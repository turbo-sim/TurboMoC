
/* Browser-only display copies and widths. CFD coordinates remain immutable. */
window.dash_clientside = Object.assign({}, window.dash_clientside, {
  studio: {
    scenes: {},
    roots: {},
    element: function(description) {
      const props = Object.assign({}, description.props);
      const children = props.children || [];
      delete props.children;
      // These are the installed dash-vtk React components, with the same
      // bundled VTK/shader implementation as the original Dash viewer.
      props.key = props.id || props.name || description.type;
      return React.createElement(window.dash_vtk[description.type], props,
        ...children.map(child => window.dash_clientside.studio.element(child)));
    },
    mount: function(pane) {
      const studio = window.dash_clientside.studio;
      const target = document.getElementById(pane + "-scene");
      let mounted = studio.roots[pane];
      if (!mounted) {
        const host = document.createElement("div");
        host.style.height = "100%";
        host.style.width = "100%";
        target.appendChild(host);
        // Dash remounts descendants when a serialized children property is
        // replaced, even with stable IDs. Own this small React subtree so
        // camera/context lifetime follows React reconciliation instead.
        const root = ReactDOM.createRoot ? ReactDOM.createRoot(host)
          : {render: element => ReactDOM.render(element, host)};
        mounted = {host, root, id: pane + "-vtk-" + Date.now()};
        studio.roots[pane] = mounted;
      } else if (mounted.host.parentElement !== target) {
        target.appendChild(mounted.host);
      }
      return mounted;
    },
    commit: function(scene) {
      const studio = window.dash_clientside.studio;
      if (studio.scenes[scene.pane] !== scene) return;
      scene.root.render(studio.view(scene));
    },
    updateMeshStyle: function(scene) {
      const studio = window.dash_clientside.studio;
      if (!scene.ready || !scene.active || studio.scenes[scene.pane] !== scene) return;
      const element = document.getElementById(scene.id);
      const canvas = element && element.querySelector("canvas");
      const size = canvas && canvas.getBoundingClientRect();
      const pixelRatio = size && size.width > 0 ? canvas.width / size.width : 1;
      const styleKey = [scene.width, scene.edges, pixelRatio].join("|");
      if (scene.styleKey === styleKey) return;
      scene.styleKey = styleKey;
      for (const actor of scene.actors) {
        if (!actor.props.actor.visibility) continue;
        actor.props.mapper = window.turbomocMeshEdges.mapper(scene.width, actor.displayEdges, pixelRatio);
      }
      studio.commit(scene);
      // Wait for React to apply mapper settings before requesting a new frame.
      requestAnimationFrame(() => {
        if (studio.scenes[scene.pane] === scene && scene.active) {
          scene.viewProps.triggerRender = Date.now();
          studio.commit(scene);
        }
      });
    },
    fit: function(id, key, pane, count, attempt) {
      const studio = window.dash_clientside.studio;
      const scene = studio.scenes[pane];
      if (!scene || !scene.active || scene.key !== key) return;
      const el = document.getElementById(id);
      if (el && el.querySelectorAll("[id^='" + pane + "-poly-']").length === count) {
        if (pane === "mesh") {
          // dash-vtk mounts point arrays before applying the parent's mapper
          // options. Its bundled vtk.js omits custom attributes from the VBO
          // cache key. Refresh the edge arrays once after mount, then enable
          // the shader only when their GPU buffers have been rebuilt.
          for (const actor of scene.actors) {
            if (!actor.props.actor.visibility) continue;
            const array = actor.props.children[0].props.children[0].props.children[0];
            array.props.values = array.props.values.slice();
          }
          studio.commit(scene);
          requestAnimationFrame(() => {
            if (studio.scenes[pane] !== scene || !scene.active) return;
            scene.ready = true;
            studio.updateMeshStyle(scene);
          });
        }
        // A dataset can finish while its workspace is hidden. Fit on the first
        // visible resize, then preserve the user's camera on later resizes.
        const observer = new ResizeObserver(entries => {
          if (studio.scenes[pane] !== scene || !scene.active) return;
          const size = entries[0].contentRect;
          if (pane === "mesh" && size.width > 0 && size.height > 0)
            requestAnimationFrame(() => studio.updateMeshStyle(scene));
          if (size.width > 0 && size.height > 0 && !scene.fitted) {
            scene.fitted = true;
            setTimeout(() => {
              if (studio.scenes[pane] === scene && scene.active && document.getElementById(id)) {
                scene.viewProps.triggerResetCamera = Date.now();
                scene.viewProps.triggerRender = Date.now();
                studio.commit(scene);
              }
            }, 350);
          }
        });
        scene.observer = observer;
        observer.observe(el);
      } else if (attempt < 100) {
        setTimeout(() => studio.fit(id, key, pane, count, attempt + 1), 50);
      }
    },
    view: function(scene) {
      const studio = window.dash_clientside.studio;
      return React.createElement(React.Suspense, {fallback: null},
        React.createElement(window.dash_vtk.View,
          Object.assign({}, scene.viewProps, {key: scene.id, id: scene.id,
            background: [.98,.985,.987], style: {width: "100%", height: "100%"},
            cameraPosition: [1,1,1], cameraViewUp: [0,0,1]}),
          ...scene.actors.map(actor => studio.element(actor))));
    },
    hideActors: function(actors) {
      return actors.map(actor => Object.assign({}, actor, {props: Object.assign({}, actor.props, {
        actor: Object.assign({}, actor.props.actor, {visibility: false})
      })}));
    },
    render: function(data, settings, pane, plottedRows, width) {
      const studio = window.dash_clientside.studio;
      const previous = studio.scenes[pane];
      settings = settings || {};
      const rows = data && (data.rows || [data]).filter(row =>
        plottedRows == null || plottedRows.includes(row.id || 'stator_1'));
      if (!rows || !rows.length) {
        if (previous && previous.observer) previous.observer.disconnect();
        const note = plottedRows && !plottedRows.length ? "Select a row under Show to display it."
                   : data && data.rows && data.rows.length ? "No generated mesh or geometry for the shown rows."
                   : pane === "mesh" ? "Generate a current 3D mesh to show its boundary cells."
                                    : "Compute current MoC to show geometry.";
        if (!previous) return [window.dash_clientside.no_update, note];
        const hidden = Object.assign({}, previous, {key: "empty", active: false,
          ready: false, observer: null, actors: studio.hideActors(previous.actors)});
        studio.scenes[pane] = hidden;
        // Retain the View even while no current artifact is selected. Its
        // bundled timers and camera listeners then always reference live VTK
        // objects; stale data are hidden rather than drawn as current results.
        studio.commit(hidden);
        return [window.dash_clientside.no_update, note];
      }
      const lineWidth = Number(width) > 0 ? Number(width) : 0.35;
      // Visibility changes actor selection, but retains the numerical data key
      // and camera. No regeneration or removal of hidden row results is needed.
      const key = [data.key, JSON.stringify(rows.map(row => [row.id, settings[row.id]]))].join("|");
      const note = rows.map(row => (row.label || row.id) + ": " +
        Math.min(row.is_linear ? 25 : row.num_blades, Math.max(1, Math.round((settings[row.id] || {}).copies || 1)))
        + " displayed").join(" · ") + " · XYZ [mm] · " +
        (pane === "mesh" ? "actual volume boundary cells" : "geometry display tessellation") +
        ". Drag to rotate; scroll to zoom.";
      if (previous && previous.key === key) {
        // Width and edge visibility change only the shader. Reuse the actors,
        // their data and the camera; no numerical job or geometry transfer.
        if (pane === "mesh") {
          previous.width = lineWidth;
          studio.updateMeshStyle(previous);
        }
        return [window.dash_clientside.no_update, note];
      }
      if (previous && previous.observer) previous.observer.disconnect();
      const colors = {blade: [1,.55,.08], hub: [.65,.70,.72], shroud: [.8,.82,.84],
        inlet: [.4,.6,.75], outlet: [.45,.65,.55], periodic_left: [.7,.6,.75],
        periodic_right: [.75,.71,.52]};
      const actors = [];
      // Row datasets remain independent. Translation/rotation is a display
      // transform, so changing assembly placement never mutates mesh nodes.
      for (const row of rows) {
        const display = settings[row.id] || {};
        const count = Math.max(1, Math.round(display.copies || 1));
        const names = display.surfaces || ["blade", "hub"];
        const rowCount = row.is_linear ? count : Math.min(row.num_blades, count);
        const placement = row.placement || {};
        const rowId = row.id || 'stator_1';
        for (let i = 0; i < rowCount; ++i) for (const name of names) {
        const original = row.surfaces[name];
        if (!original) continue;
        const s = pane === "mesh" ? window.turbomocMeshEdges.prepare(original) : original;
        const arrays = [];
        if (s.normals) arrays.push({namespace: "dash_vtk", type: "DataArray", props: {
            name: "Normals", registration: "setNormals", numberOfComponents: 3,
            type: "Float32Array", values: s.normals}});
        if (s.coordinates) arrays.push({namespace: "dash_vtk", type: "DataArray", props: {
            id: pane + "-edge-coordinates-" + rowId + "-" + i + "-" + name,
            name: "MeshEdgeCoordinates", numberOfComponents: 3,
            type: "Float32Array", values: s.coordinates}});
        const fields = arrays.length ? [{namespace: "dash_vtk", type: "PointData",
          props: {children: arrays}}] : [];
        const hex = (display.colors || {})[name];
        const color = /^#[0-9a-f]{6}$/i.test(hex || "") ?
          [1, 3, 5].map(offset => parseInt(hex.slice(offset, offset + 2), 16) / 255) : colors[name];
        actors.push({namespace: "dash_vtk", type: "GeometryRepresentation",
          displayEdges: !!display.edges, props: {
          id: pane + "-actor-" + rowId + "-" + i + "-" + name,
          actor: {visibility: true,
            orientation: [0,0,row.is_linear ? 0 : (placement.angular_phase_deg || 0) - 360 * i / row.num_blades],
            position: row.is_linear ? [0, i * row.pitch_mm, 0] : [0, 0, placement.streamwise_offset_mm || 0]},
          mapper: pane === "mesh" ? window.turbomocMeshEdges.mapper(lineWidth, false)
                                  : {scalarVisibility: false},
          property: {color, edgeColor: [0,0,0],
            edgeVisibility: false, ambient: .25, diffuse: .75,
            interpolation: pane === "geometry" ? 2 : 1},
          children: [{namespace: "dash_vtk", type: "PolyData", props: {
            id: pane + "-poly-" + rowId + "-" + i + "-" + name, points: s.points, polys: s.polys,
            connectivity: "manual", children: fields}}] }});
      }
      }
      // dash-vtk 0.0.9 does not cancel its mount-time camera timer or debounced
      // bounds callback on unmount. Its representation cube axes also attach
      // camera listeners. Keep one View and a stable actor pool for this pane:
      // update existing datasets in place and hide unused copies/surfaces.
      // Stable order and IDs let React retain each representation's lifetime.
      const desired = new Map(actors.map(actor => [actor.props.id, actor]));
      const pool = [];
      if (previous) {
        for (const old of previous.actors) {
          const current = desired.get(old.props.id);
          pool.push(current || studio.hideActors([old])[0]);
          desired.delete(old.props.id);
        }
      }
      pool.push(...desired.values());
      const mounted = studio.mount(pane);
      const id = mounted.id;
      const fitted = !!(previous && previous.dataKey === data.key && previous.fitted);
      const scene = {id, pane, root: mounted.root, viewProps: previous ? previous.viewProps : {},
        key, dataKey: data.key, width: lineWidth,
        edges: JSON.stringify(rows.map(row => !!(settings[row.id] || {}).edges)),
        actors: pool, active: actors.length > 0, ready: false, fitted};
      studio.scenes[pane] = scene;
      studio.commit(scene);
      // Fit new numerical datasets; display controls preserve the camera.
      if (scene.active) setTimeout(() => studio.fit(id, key, pane, pool.length, 0), 0);
      return [window.dash_clientside.no_update,
              actors.length ? note : "Select a surface to display."];
    }
  }
});
// Native details elements keep their open state. Notify the renderers when a
// collapsed plot reopens; data and numerical artifacts do not change.
document.addEventListener("toggle", event => {
  if (event.target.tagName === "DETAILS" && event.target.open)
    window.dispatchEvent(new Event("resize"));
}, true);

// Native color controls feed Dash stores without adding a UI dependency.
function mountSurfaceColors() {
  document.querySelectorAll('.surface-color-picker').forEach(host => {
    let picker = host.querySelector('input');
    if (!picker) {
      picker = document.createElement('input');
      picker.type = 'color';
      picker.addEventListener('input', () => {
        window.dash_clientside.set_props(JSON.parse(host.dataset.colorId), {data: picker.value});
      });
      host.appendChild(picker);
    }
    picker.setAttribute('aria-label', host.dataset.colorLabel);
    picker.title = host.dataset.colorLabel;
    if (picker.value !== host.dataset.color) picker.value = host.dataset.color;
  });
}
new MutationObserver(mountSurfaceColors).observe(document.documentElement,
  {subtree: true, childList: true, attributes: true, attributeFilter: ['data-color']});
mountSurfaceColors();

