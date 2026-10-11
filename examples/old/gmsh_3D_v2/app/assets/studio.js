
/* Browser-only display copies and widths. CFD coordinates remain immutable. */
window.dash_clientside = Object.assign({}, window.dash_clientside, {
  studio: {
    scenes: {},
    updateMeshStyle: function(scene) {
      if (!scene.ready) return;
      const element = document.getElementById(scene.id);
      const canvas = element && element.querySelector("canvas");
      const size = canvas && canvas.getBoundingClientRect();
      const pixelRatio = size && size.width > 0 ? canvas.width / size.width : 1;
      const styleKey = [scene.width, scene.edges, pixelRatio].join("|");
      if (scene.styleKey === styleKey) return;
      scene.styleKey = styleKey;
      const mapper = window.turbomocMeshEdges.mapper(scene.width, scene.edges, pixelRatio);
      for (const actor of scene.actors) {
        actor.props.mapper = mapper;
        window.dash_clientside.set_props(actor.props.id, {mapper});
      }
      // Wait for React to apply mapper settings before requesting a new frame.
      requestAnimationFrame(() => window.dash_clientside.set_props(scene.id,
        {triggerRender: Date.now()}));
    },
    fit: function(id, key, pane, count, attempt) {
      const studio = window.dash_clientside.studio;
      const scene = studio.scenes[pane];
      if (!scene || scene.key !== key) return;
      const el = document.getElementById(id);
      if (el && el.querySelectorAll("[id^='" + pane + "-poly-']").length === count) {
        if (pane === "mesh") {
          // dash-vtk mounts point arrays before applying the parent's mapper
          // options. Its bundled vtk.js omits custom attributes from the VBO
          // cache key. Refresh the edge arrays once after mount, then enable
          // the shader only when their GPU buffers have been rebuilt.
          for (const actor of scene.actors) {
            const array = actor.props.children[0].props.children[0].props.children[0];
            window.dash_clientside.set_props(array.props.id,
              {values: array.props.values.slice()});
          }
          requestAnimationFrame(() => {
            if (studio.scenes[pane] !== scene) return;
            scene.ready = true;
            studio.updateMeshStyle(scene);
          });
        }
        // A dataset can finish while its workspace is hidden. Fit on the first
        // visible resize, then preserve the user's camera on later resizes.
        const observer = new ResizeObserver(entries => {
          const size = entries[0].contentRect;
          if (pane === "mesh" && size.width > 0 && size.height > 0)
            requestAnimationFrame(() => studio.updateMeshStyle(scene));
          if (size.width > 0 && size.height > 0 && !scene.fitted) {
            scene.fitted = true;
            setTimeout(() => {
              if (studio.scenes[pane] === scene && document.getElementById(id))
                window.dash_clientside.set_props(id,
                  {triggerResetCamera: Date.now(), triggerRender: Date.now()});
            }, 350);
          }
        });
        scene.observer = observer;
        observer.observe(el);
      } else if (attempt < 100) {
        setTimeout(() => studio.fit(id, key, pane, count, attempt + 1), 50);
      }
    },
    render: function(data, copies, selected, edges, width, pane) {
      const studio = window.dash_clientside.studio;
      const previous = studio.scenes[pane];
      if (!data) {
        if (previous && previous.observer) previous.observer.disconnect();
        studio.scenes[pane] = null;
        return [[], "Compute current MoC to show geometry."];
      }
      const count = Math.min(data.num_blades, Math.max(1, Math.round(copies || 1)));
      const names = (selected || []).filter(n => data.surfaces[n]);
      const showEdges = pane === "mesh" && (edges || []).includes("edges");
      const lineWidth = Number(width) > 0 ? Number(width) : 0.35;
      const key = [data.key, count, names.join(",")].join("|");
      const note = count + " displayed blade passage(s) · XYZ [mm] · " +
        (pane === "mesh" ? "actual volume boundary cells" : "geometry display tessellation") +
        ". Drag to rotate; scroll to zoom.";
      if (previous && previous.key === key) {
        // Width and edge visibility change only the shader. Reuse the actors,
        // their data and the camera; no numerical job or geometry transfer.
        if (pane === "mesh") {
          previous.width = lineWidth;
          previous.edges = showEdges;
          studio.updateMeshStyle(previous);
        }
        return [window.dash_clientside.no_update, note];
      }
      if (previous && previous.observer) previous.observer.disconnect();
      const colors = {blade: [1,.55,.08], hub: [.65,.70,.72], shroud: [.8,.82,.84],
        inlet: [.4,.6,.75], outlet: [.45,.65,.55], periodic_left: [.7,.6,.75],
        periodic_right: [.75,.71,.52]};
      const actors = [];
      for (let i = 0; i < count; ++i) for (const name of names) {
        const original = data.surfaces[name];
        const s = pane === "mesh" ? window.turbomocMeshEdges.prepare(original) : original;
        const arrays = [];
        if (s.normals) arrays.push({namespace: "dash_vtk", type: "DataArray", props: {
            name: "Normals", registration: "setNormals", numberOfComponents: 3,
            type: "Float32Array", values: s.normals}});
        if (s.coordinates) arrays.push({namespace: "dash_vtk", type: "DataArray", props: {
            id: pane + "-edge-coordinates-" + i + "-" + name,
            name: "MeshEdgeCoordinates", numberOfComponents: 3,
            type: "Float32Array", values: s.coordinates}});
        const fields = arrays.length ? [{namespace: "dash_vtk", type: "PointData",
          props: {children: arrays}}] : [];
        actors.push({namespace: "dash_vtk", type: "GeometryRepresentation", props: {
          id: pane + "-actor-" + i + "-" + name,
          actor: {orientation: [0,0,-360 * i / data.num_blades]},
          mapper: pane === "mesh" ? window.turbomocMeshEdges.mapper(lineWidth, false)
                                  : {scalarVisibility: false},
          property: {color: colors[name], edgeColor: [0,0,0],
            edgeVisibility: false, ambient: .25, diffuse: .75,
            interpolation: pane === "geometry" ? 2 : 1},
          children: [{namespace: "dash_vtk", type: "PolyData", props: {
            id: pane + "-poly-" + i + "-" + name, points: s.points, polys: s.polys,
            connectivity: "manual", children: fields}}] }});
      }
      // dash-vtk 0.0.9 needs a fresh View for a changed actor set: otherwise
      // camera listeners may stay attached to deleted representations.
      const id = pane + "-vtk-" + Date.now();
      const view = {namespace: "dash_vtk", type: "View", props: {
        id, background: [.98,.985,.987], style: {width: "100%", height: "100%"},
        cameraPosition: [1,1,1], cameraViewUp: [0,0,1], children: actors}};
      const wrapper = {namespace: "dash_html_components", type: "Div", props: {
        key, style: {height: "100%"}, children: [view]}};
      studio.scenes[pane] = {id, key, width: lineWidth, edges: showEdges, actors, fitted: false};
      if (actors.length) setTimeout(() => studio.fit(id, key, pane, actors.length, 0), 0);
      return [[wrapper], note];
    }
  }
});
// Native details elements keep their open state. Notify the renderers when a
// collapsed plot reopens; data and numerical artifacts do not change.
document.addEventListener("toggle", event => {
  if (event.target.tagName === "DETAILS" && event.target.open)
    window.dispatchEvent(new Event("resize"));
}, true);

