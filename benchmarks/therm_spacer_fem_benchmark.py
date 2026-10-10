"""Independent conforming-FEM check of LBNL's public THERM spacer case.

The source is ``examples/products/Spacer_CMA.thmx`` in LBNL-ETA/pyWinCalc.
This parses public polygons, materials, boundary segments, and the stored
U-factor, but does not invoke THERM. Material polygons are ear-clipped and
uniformly refined; coincident nodes are merged across interfaces.
"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

NS = {"t": "http://windows.lbl.gov"}


def xml_points(item: ET.Element) -> np.ndarray:
    return np.array([[float(p.attrib["x"]), float(p.attrib["y"])]
                     for p in item.findall("t:Point", NS)]) * 1e-3


def parse_case(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    materials = {item.attrib["Name"]: float(item.attrib["Conductivity"])
                 for item in root.findall("t:Materials/t:Material", NS)}
    polygons = [(item.attrib["Material"], xml_points(item))
                for item in root.findall("t:Polygons/t:Polygon", NS)]
    bc_data = {item.attrib["Name"]: {
        "h": float(item.attrib["H"]),
        "temperature": float(item.attrib["Temperature"]),
    } for item in root.findall("t:BoundaryConditions/t:BoundaryCondition", NS)}
    boundaries = [{
        "name": item.attrib["BC"],
        "tag": item.attrib.get("UFactorTag", ""),
        "points": xml_points(item),
    } for item in root.findall("t:Boundaries/t:BCPolygon", NS)]
    reference = reference_length = None
    for ufactors in root.findall("t:Results/t:Case/t:U-factors", NS):
        tag = ufactors.find("t:Tag", NS)
        if tag is None or tag.text != "SpacerInteriorTag":
            continue
        for projection in ufactors.findall("t:Projection", NS):
            kind = projection.find("t:Length-type", NS)
            if kind is not None and kind.text == "Total length":
                reference = float(projection.find("t:U-factor", NS).attrib["value"])
                reference_length = float(projection.find("t:Length", NS).attrib["value"]) * 1e-3
    if reference is None or reference_length is None:
        raise ValueError("THERM reference result was not found")
    return {"materials": materials, "polygons": polygons, "bc_data": bc_data,
            "boundaries": boundaries, "reference_u": reference,
            "reference_length": reference_length,
            "therm_version": root.find("t:ThermVersion", NS).text}


def signed_area(vertices: np.ndarray) -> float:
    x, y = vertices.T
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def point_in_triangle(point: np.ndarray, triangle: np.ndarray, tol: float = 1e-14) -> bool:
    a, b, c = triangle
    cross = lambda u, v: u[0] * v[1] - u[1] * v[0]
    denominator = cross(b - a, c - a)
    first = cross(b - a, point - a) / denominator
    second = cross(point - a, c - a) / denominator
    return first >= -tol and second >= -tol and first + second <= 1.0 + tol


def triangulate_polygon(vertices: np.ndarray) -> list[np.ndarray]:
    """Ear-clip a simple polygon while retaining exact interface vertices."""
    vertices = vertices.copy()
    if signed_area(vertices) < 0:
        vertices = vertices[::-1]
    indices = list(range(len(vertices)))
    triangles: list[np.ndarray] = []
    while len(indices) > 3:
        for position, current in enumerate(indices):
            previous = indices[position - 1]
            following = indices[(position + 1) % len(indices)]
            a, b, c = vertices[[previous, current, following]]
            cross_product = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            if cross_product <= 1e-16:
                continue
            triangle = np.array([a, b, c])
            if any(point_in_triangle(vertices[index], triangle)
                   for index in indices
                   if index not in (previous, current, following)):
                continue
            triangles.append(triangle)
            del indices[position]
            break
        else:
            raise ValueError("Could not triangulate THERM polygon")
    triangles.append(vertices[indices])
    return triangles


def refine(triangles: list[tuple[np.ndarray, float]], levels: int) -> list[tuple[np.ndarray, float]]:
    for _ in range(levels):
        refined = []
        for triangle, conductivity in triangles:
            a, b, c = triangle
            ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
            refined.extend([(np.array([a, ab, ca]), conductivity),
                            (np.array([ab, b, bc]), conductivity),
                            (np.array([ca, bc, c]), conductivity),
                            (np.array([ab, bc, ca]), conductivity)])
        triangles = refined
    return triangles


def subdivide_segment(points: np.ndarray, levels: int) -> list[tuple[np.ndarray, np.ndarray]]:
    segments = [(points[0], points[1])]
    for _ in range(levels):
        next_segments = []
        for a, b in segments:
            midpoint = (a + b) / 2
            next_segments.extend([(a, midpoint), (midpoint, b)])
        segments = next_segments
    return segments


def solve_case(case: dict[str, object], levels: int) -> dict[str, float]:
    base: list[tuple[np.ndarray, float]] = []
    for material, vertices in case["polygons"]:
        base.extend((triangle, case["materials"][material])
                    for triangle in triangulate_polygon(vertices))
    triangles = refine(base, levels)
    nodes: list[np.ndarray] = []
    lookup: dict[tuple[float, float], int] = {}

    def node_id(point: np.ndarray) -> int:
        key = tuple(np.round(point, 12))
        if key not in lookup:
            lookup[key] = len(nodes)
            nodes.append(point)
        return lookup[key]

    elements = [(np.array([node_id(point) for point in triangle]), conductivity)
                for triangle, conductivity in triangles]
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    # Boundary refinement cannot create nodes absent from refined triangles.
    boundary_segments = []
    for boundary in case["boundaries"]:
        for a, b in subdivide_segment(boundary["points"], levels):
            boundary_segments.append((boundary, node_id(a), node_id(b),
                                      float(np.linalg.norm(b - a))))
    rhs = np.zeros(len(nodes))

    for element, conductivity in elements:
        coordinates = np.array([nodes[index] for index in element])
        x, y = coordinates.T
        area = abs(np.linalg.det(np.array([[1.0, x[0], y[0]],
                                           [1.0, x[1], y[1]],
                                           [1.0, x[2], y[2]]]))) / 2.0
        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
        local = conductivity * (np.outer(b, b) + np.outer(c, c)) / (4.0 * area)
        for local_i, global_i in enumerate(element):
            for local_j, global_j in enumerate(element):
                rows.append(global_i); cols.append(global_j); data.append(local[local_i, local_j])

    indoor_terms = []
    for boundary, ia, ib, length in boundary_segments:
        bc = case["bc_data"][boundary["name"]]
        if bc["h"] == 0:
            continue
        h = bc["h"]
        for i, j, coefficient in ((ia, ia, 2), (ia, ib, 1),
                                  (ib, ia, 1), (ib, ib, 2)):
            rows.append(i); cols.append(j); data.append(h * length * coefficient / 6.0)
        rhs[ia] += h * bc["temperature"] * length / 2.0
        rhs[ib] += h * bc["temperature"] * length / 2.0
        if boundary["tag"] == "SpacerInteriorTag":
            indoor_terms.append((ia, ib, length, h, bc["temperature"]))

    matrix = coo_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes))).tocsr()
    temperature = spsolve(matrix, rhs)
    indoor_heat = sum(h * length * (tb - (temperature[ia] + temperature[ib]) / 2.0)
                      for ia, ib, length, h, tb in indoor_terms)
    indoor = case["bc_data"]["Spacer Indoor"]
    outdoor = case["bc_data"]["Spacer Outdoor"]
    u_value = indoor_heat / (case["reference_length"] *
                             (indoor["temperature"] - outdoor["temperature"]))
    return {"refinement_level": levels, "nodes": len(nodes),
            "triangles": len(triangles), "u_W_m2K": float(u_value),
            "relative_error": float((u_value - case["reference_u"]) / case["reference_u"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("therm_xml", type=Path)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    case = parse_case(args.therm_xml)
    rows = [solve_case(case, level) for level in (1, 2, 3, 4, 5)]
    mesh_change = abs(rows[-1]["u_W_m2K"] - rows[-2]["u_W_m2K"]) / rows[-1]["u_W_m2K"]
    result = {
        "source": "LBNL-ETA/pyWinCalc examples/products/Spacer_CMA.thmx",
        "source_url": "https://github.com/LBNL-ETA/pyWinCalc/blob/main/examples/products/Spacer_CMA.thmx",
        "therm_version": case["therm_version"],
        "therm_reference_u_W_m2K": case["reference_u"],
        "therm_tagged_length_m": case["reference_length"],
        "rows": rows,
        "finest_absolute_relative_error": abs(rows[-1]["relative_error"]),
        "finest_mesh_change": mesh_change,
        "acceptance": {
            "reference_error_below_2_percent": abs(rows[-1]["relative_error"]) < 0.02,
            "finest_mesh_change_below_1_percent": mesh_change < 0.01,
        },
        "interpretation": "Independent conforming linear-triangle FEM reproduction of a public THERM polygon case.",
    }
    if args.output_directory:
        args.output_directory.mkdir(parents=True, exist_ok=True)
        (args.output_directory / "therm_spacer_benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
        with (args.output_directory / "therm_spacer_benchmark.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
