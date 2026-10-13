/* Antialiased cell edges drawn in the surface shader, not as WebGL lines.
 * Native GL lines are commonly clamped to one pixel. Here the distance to
 * each cell edge is measured in framebuffer pixels, so fractional widths
 * work at every zoom level and the surface's depth test hides back edges.
 */
window.turbomocMeshEdges = (() => {
  const cache = new WeakMap();

  // Match the inspection legend. Values are normalized by the server using
  // the metric's good/bad convention, never the currently selected subset.
  function qualityColor(value) {
    if (value == null || !Number.isFinite(Number(value))) return [137, 148, 156];
    const stops = [[35, 155, 86], [244, 208, 63], [192, 57, 43]];
    const x = Math.max(0, Math.min(1, Number(value))) * 2;
    const segment = Math.min(1, Math.floor(x));
    const fraction = x - segment;
    return stops[segment].map((component, channel) =>
      Math.round(component + fraction * (stops[segment + 1][channel] - component)));
  }

  function prepare(surface) {
    if (cache.has(surface)) return cache.get(surface);
    let triangleCount = 0, faceCount = 0;
    for (let offset = 0; offset < surface.polys.length;) {
      const size = surface.polys[offset];
      if (!Number.isInteger(size) || size < 3 || offset + size >= surface.polys.length)
        throw new Error("Mesh edge rendering expects complete polygon faces.");
      triangleCount += size - 2;
      ++faceCount;
      offset += size + 1;
    }
    for (const field of ["cell_values", "cell_ids", "color_values"]) {
      if (surface[field] != null && surface[field].length !== faceCount)
        throw new Error("Mesh inspection " + field + " must have one value per face.");
    }
    const points = new Array(triangleCount * 9);
    const coordinates = new Array(triangleCount * 9);
    const polys = new Array(triangleCount * 4);
    const cellValues = surface.cell_values == null ? null : new Array(triangleCount);
    const cellIds = surface.cell_ids == null ? null : new Array(triangleCount);
    const colors = surface.color_values == null ? null : new Array(triangleCount * 3);
    let triangle = 0, face = 0;
    for (let offset = 0; offset < surface.polys.length;) {
      const size = surface.polys[offset];
      const ids = surface.polys.slice(offset + 1, offset + 1 + size);
      const color = colors && qualityColor(surface.color_values[face]);
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
        // Every displayed triangle retains its original volume-cell value.
        // CellData avoids averaging values at shared vertices or between cells.
        if (cellValues) cellValues[triangle] = surface.cell_values[face];
        if (cellIds) cellIds[triangle] = surface.cell_ids[face];
        if (colors) for (let channel = 0; channel < 3; ++channel)
          colors[triangle * 3 + channel] = color[channel];
        ++triangle;
      }
      ++face;
      offset += size + 1;
    }
    const result = {points, polys, coordinates, cellValues, cellIds, colors};
    cache.set(surface, result);
    return result;
  }

  function mapper(width, enabled, pixelRatio = 1, scalarOptions = {}) {
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
    return {scalarVisibility: false, ...scalarOptions, customShaderAttributes: ["MeshEdgeCoordinates"],
      viewSpecificProperties: {OpenGL: {ShaderReplacements: replacements}}};
  }

  return {prepare, mapper, qualityColor};
})();
