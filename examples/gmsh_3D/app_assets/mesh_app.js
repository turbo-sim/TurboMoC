/* VTK datasets are sent once. Rotation copies and surface toggles stay in the
 * browser, avoiding repeated transfers of large mesh-coordinate arrays. */
window.dash_clientside = Object.assign({}, window.dash_clientside, {
  mesh_app: {
    renderScene: function (geometry, mesh, state, tab, requestedCount) {
      const is2D = tab === "conformal";
      const panel2D = {display: is2D ? "block" : "none"};
      const panel3D = {display: is2D ? "none" : "block", height: "64vh"};
      const data = tab === "mesh" ? mesh : geometry;
      if (!data || !state || !state.config) {
        return [[], panel2D, panel3D,
          tab === "mesh" ? "Create a mesh to inspect it here." : "Compute MoC to preview the geometry."];
      }
      const options = state.config.mesh.pyvista || {};
      const selection = options.surfaces || {blade: true};
      const names = Object.keys(selection).filter(name => selection[name] && data.surfaces[name]);
      const count = Math.min(Math.max(1, requestedCount || 1), data.num_blades);
      const colors = {blade: [1,.55,0], hub: [.7,.7,.7], shroud: [.85,.85,.85],
        inlet: [.39,.58,.93], outlet: [.24,.7,.44], periodic_left: [.87,.63,.87],
        periodic_right: [.94,.9,.55]};
      const sceneKey = [tab, data.geometry_key || data.mesh_key,
        data.solution_job, count, names.join(","), !!options.show_mesh_edges].join("|");
      const unchanged = window.dash_clientside.mesh_app.lastSceneKey === sceneKey;
      window.dash_clientside.mesh_app.lastSceneKey = sceneKey;
      const representations = [];
      if (!is2D) {
        for (let index = 0; index < count; index++) {
          for (const name of names) {
            const surface = data.surfaces[name];
            representations.push({namespace: "dash_vtk", type: "GeometryRepresentation", props: {
              id: "surface-" + index + "-" + name,
              actor: {orientation: [0, 0, -360 * index / data.num_blades]},
              mapper: {scalarVisibility: false},
              property: {color: colors[name], edgeColor: [0,0,0], opacity: 1,
                edgeVisibility: tab === "mesh" && !!options.show_mesh_edges},
              children: [{namespace: "dash_vtk", type: "PolyData", props: {
                points: surface.points, polys: surface.polys, connectivity: "manual"
              }}]
            }});
          }
        }
      }
      let note = is2D ? "Conformal plane: θ in radians and dimensionless m′." :
        "3D coordinates in mm. Drag to rotate; scroll to zoom. Displaying " + count + " blade passage(s).";
      if (tab === "mesh") {
        note += " " + data.cells.toLocaleString() + " cells; minimum SICN " + data.min_sicn.toFixed(4) + ".";
        if (data.mesh_key !== state.mesh_key) {
          note += " This mesh represents an earlier configuration. Click Create mesh to update it.";
        }
      }
      // Remount the complete view when its actor set changes. Removing actors
      // from dash-vtk 0.0.9 can leave camera listeners attached to deleted axes.
      // A fresh view owns its camera and avoids those stale subscriptions.
      const view = {namespace: "dash_vtk", type: "View", props: {
        id: "vtk-scene-" + Date.now(),
        background: [.96, .95, .92],
        style: {width: "100%", height: "100%"},
        cameraPosition: [1, 1, 1], cameraViewUp: [0, 0, 1],
        children: representations
      }};
      const container = {namespace: "dash_html_components", type: "Div", props: {
        key: sceneKey, style: {height: "100%"}, children: [view]
      }};
      // Let the visible pane settle before VTK resizes its canvas.
      setTimeout(() => window.dispatchEvent(new Event("resize")), 80);
      return [unchanged ? window.dash_clientside.no_update : [container],
        panel2D, panel3D, note];
    }
  }
});
