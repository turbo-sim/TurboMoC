/* Antialiased cell edges drawn in the surface shader, not as WebGL lines.
 * Native GL lines are commonly clamped to one pixel. Here the distance to
 * each cell edge is measured in framebuffer pixels, so fractional widths
 * work at every zoom level and the surface's depth test hides back edges.
 */
window.turbomocMeshEdges = (() => {
  const cache = new WeakMap();

  function prepare(surface) {
    if (cache.has(surface)) return cache.get(surface);
    let triangleCount = 0;
    for (let offset = 0; offset < surface.polys.length;) {
      const size = surface.polys[offset];
      if (size !== 3 && size !== 4)
        throw new Error("Mesh edge rendering expects triangular or quadrilateral CFD faces.");
      triangleCount += size - 2;
      offset += size + 1;
    }
    const points = new Array(triangleCount * 9);
    const coordinates = new Array(triangleCount * 9);
    const polys = new Array(triangleCount * 4);
    let triangle = 0;
    for (let offset = 0; offset < surface.polys.length;) {
      const size = surface.polys[offset];
      const ids = surface.polys.slice(offset + 1, offset + 1 + size);
      for (let j = 1; j < size - 1; ++j) {
        const vertices = [ids[0], ids[j], ids[j + 1]];
        // Barycentric component k vanishes on the opposite triangle edge.
        // Set a component to one throughout the triangle to hide an internal
        // triangulation diagonal: only original CFD cell edges are drawn.
        const visible = [true, j === size - 2, j === 1];
        for (let vertex = 0; vertex < 3; ++vertex) {
          for (let axis = 0; axis < 3; ++axis) {
            const index = triangle * 9 + vertex * 3 + axis;
            points[index] = surface.points[vertices[vertex] * 3 + axis];
            coordinates[index] = visible[axis] ? Number(vertex === axis) : 1;
          }
        }
        polys[triangle * 4] = 3;
        for (let vertex = 0; vertex < 3; ++vertex)
          polys[triangle * 4 + vertex + 1] = triangle * 3 + vertex;
        ++triangle;
      }
      offset += size + 1;
    }
    const result = {points, polys, coordinates};
    cache.set(surface, result);
    return result;
  }

  function mapper(width, enabled, pixelRatio = 1) {
    const replacements = [];
    function replace(shaderType, originalValue, code) {
      replacements.push({shaderType, originalValue, replaceFirst: true,
        replacementValue: originalValue + "\n" + code, replaceAll: false});
    }
    if (enabled) {
      replace("Vertex", "//VTK::PositionVC::Dec",
        "attribute vec3 MeshEdgeCoordinatesMC;\nvarying vec3 meshEdgeCoordinates;");
      replace("Vertex", "//VTK::PositionVC::Impl",
        "meshEdgeCoordinates = MeshEdgeCoordinatesMC;");
      replace("Fragment", "//VTK::PositionVC::Dec",
        "varying vec3 meshEdgeCoordinates;");
      // |gradient(b)| converts a barycentric coordinate b into pixel distance.
      // The difference of two ramps integrates a one-pixel box filter over
      // the stroke: a 0.35 px edge gets fractional coverage, not a 1 px clamp.
      const halfWidth = (0.5 * width * pixelRatio).toFixed(6);
      replace("Fragment", "//VTK::Light::Impl", `
        vec3 dx = dFdx(meshEdgeCoordinates);
        vec3 dy = dFdy(meshEdgeCoordinates);
        vec3 distances = meshEdgeCoordinates / max(sqrt(dx*dx + dy*dy), vec3(1.0e-8));
        float distanceToEdge = min(distances.x, min(distances.y, distances.z));
        float coverage = clamp(${halfWidth} + 0.5 - distanceToEdge, 0.0, 1.0)
                       - clamp(0.5 - ${halfWidth} - distanceToEdge, 0.0, 1.0);
        gl_FragData[0].rgb *= 1.0 - coverage;
      `);
    }
    return {scalarVisibility: false, customShaderAttributes: ["MeshEdgeCoordinates"],
      viewSpecificProperties: {OpenGL: {ShaderReplacements: replacements}}};
  }

  return {prepare, mapper};
})();
