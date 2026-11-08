// TurboMoC policy and composition. Rendering, pointer interaction, camera
// navigation, numeric editing and evaluation queues stay in nurbspy.
let current = null;

function node(tag, className, text) {
    const result = document.createElement(tag);
    if (className) result.className = className;
    if (text !== undefined) result.textContent = text;
    return result;
}

function status(message = '', disabled = false, error = false, busy = disabled) {
    window.dash_clientside.set_props('cp-message', {children: message, title: message, className: error ? 'cp-error' : ''});
    window.dash_clientside.set_props('cp-apply', {disabled});
    window.dash_clientside.set_props('cp-reset', {disabled: busy});
}

function isBusy(viewer) {
    return Boolean(viewer.pending || viewer.inFlight || viewer.drag || viewer.resampling ||
        viewer.editing.box || viewer.editing.history.locked);
}

export function close() {
    current?.viewer.dispose();
    current = null;
    document.getElementById('cp-viewport')?.replaceChildren();
    return {closed: true};
}

function configurePanel(viewer, session) {
    // Reuse the already wired nurbspy controls in one pinned pane.
    const editing = viewer.editing, body = editing.panel.body;
    editing.panel.sidebar.classList.add('cp-point-panel');
    editing.panel.sidebar.querySelector('h4').textContent = 'Control point';
    editing.panel.sidebar.querySelector('.nrb-panel-heading button').hidden = true;
    const selection = editing.pointSelect.closest('details');
    selection.querySelector('summary').hidden = true;
    selection.open = true;
    editing.pointSelect.closest('label').firstChild.textContent = 'Active point';
    editing.pointSelect.title = 'Coordinates and Reset point affect the active point. Drag to move the whole selection.';
    editing.selectionCount.title = 'Shift-click to add or remove points. Drag a selected point to move the group.';
    editing.coordinates.forEach((input, axis) => {
        input.closest('label').firstChild.textContent = (axis === 0 ? 'X' : 'Y') + ' [mm]';
        if (axis === 2) input.closest('label').remove();
    });
    body.replaceChildren(selection);
    const displacement = node('div', 'cp-displacement');
    displacement.append(node('span', 'cp-panel-caption', 'From original position'));
    viewer.deltaOutputs = [0, 1].map(axis => {
        const row = node('div', 'cp-delta-row'), output = node('output', '', '0');
        output.setAttribute('aria-label', (axis === 0 ? 'X' : 'Y') + ' displacement [mm]');
        row.append(node('span', '', (axis === 0 ? 'Δx' : 'Δy') + ' [mm]'), output);
        displacement.append(row); return output;
    });
    viewer.resetPoint = node('button', 'cp-reset-point', 'Reset point');
    viewer.resetPoint.type = 'button';
    viewer.resetPoint.addEventListener('click', () => {
        if (!session.editable.includes(viewer.selected) || !editing.beginChange('reset-point')) return;
        const position = viewer.handles[viewer.selected].position.clone();
        position.set(...session.blade.control_points[viewer.selected], 0);
        editing.setPoint(viewer.selected, position);
        editing.commitChange();
    });
    displacement.append(viewer.resetPoint); body.append(displacement);
    const view = node('details', 'nrb-config-group cp-view-options'); view.open = true;
    view.append(node('summary', '', 'View options'));
    const controls = node('div', 'nrb-config-body');
    viewer.gridCheck.closest('label').firstChild.textContent = 'Grid';
    viewer.netCheck.closest('label').firstChild.textContent = 'Control polygon';
    controls.append(viewer.gridCheck.closest('label'), viewer.netCheck.closest('label'));
    const toggle = (label, checked, change) => {
        const row = node('label', 'nrb-science-row', label), input = document.createElement('input');
        input.type = 'checkbox'; input.checked = checked; input.setAttribute('aria-label', label);
        input.addEventListener('change', () => { change(input.checked); viewer.render(); });
        row.append(input); controls.append(row);
    };
    toggle('Original outline', true, visible => { viewer.baselineLine.visible = visible; });
    toggle('Adjacent blades', true, visible => { viewer.adjacentGroup.visible = visible; });
    view.append(controls); body.append(view);
    body.append(node('p', 'cp-lock-note', 'The first and last two points are fixed to preserve the trailing edge.'));
    viewer.panels.display.sidebar.remove();
    viewer.displayToggle.remove(); editing.panel.toggle.remove();
    viewer.root.querySelector('.nrb-panel-actions').remove();
    viewer.root.querySelector('.nrb-heading h3').remove(); viewer.resetButton.remove();
    viewer.root.querySelector('.nrb-heading button[title="Fit all control points in the view"]').textContent = 'Fit passage';
    const legend = node('div', 'cp-legend');
    for (const [kind, label] of [['active', 'Editing blade'], ['adjacent', 'Adjacent blades'], ['original', 'Original outline']]) {
        legend.append(node('span', 'cp-legend-' + kind, label));
    }
    viewer.root.querySelector('.nrb-heading').prepend(legend);
    viewer.setPanelOpen(true, 'edit');
}

export async function update(session, trigger, endpoint, assets) {
    if (trigger === 'cp-cancel') return close();
    if (!session) return null;
    if (trigger === 'cp-reset' && current) {
        const e = current.viewer;
        if (isBusy(e)) return {busy: true};
        const selection = e.editing.selection();
        e.editing.reset();
        e.setSelection(selection.indices, selection.active);
        return {reset: true};
    }
    if (current?.session.identity === session.identity) return {mounted: true, identity: session.identity};
    close(); status('Loading editor…', true);
    try {
        const [{Explorer, httpEvaluator}, THREE, {wideLine, lineGeometry}] = await Promise.all([
            import(assets + 'explorer.js'), import(assets + 'vendor/three.module.js'), import(assets + 'display.js'),
        ]);
        const state = {session, notice: '', viewer: null};
        const publish = () => {
            if (current !== state) return;
            const busy = isBusy(state.viewer);
            const error = state.notice;
            const count = session.editable.filter(i => state.viewer.handles[i].position.distanceToSquared(
                new THREE.Vector3(...session.blade.control_points[i], 0)) > 1e-24).length;
            status(error || (busy ? 'Updating shape…' : count + (count === 1 ? ' point modified' : ' points modified')),
                busy || state.viewer.displayQuality !== 'full', Boolean(error), busy);
        };
        class StatorExplorer extends Explorer {
            setViewMode() { super.setViewMode('xy'); }
            setPanelOpen() {
                if (this.activePanel !== 'edit') super.setPanelOpen(true, 'edit');
            }
            fit() {
                super.fit();
                if (this.is2D && this.adjacentGroup) {
                    this.camera.zoom = 1.12;
                    this.camera.updateProjectionMatrix(); this.render();
                }
            }
            // Route native feedback to the host's existing status row.
            setStatus(message, error = false) {
                if (error) state.notice = message;
                publish();
            }
            pickControlPoint(event) {
                const handle = super.pickControlPoint(event);
                return handle && this.options.editable.includes(handle.userData.index) ? handle : null;
            }
            pointLabel(index) {
                return this.options.editable.includes(index) ? String(index) : index + ' (fixed)';
            }
            setSelection(indices, active = indices.at(-1)) {
                // Commit a focused coordinate field to its original point
                // before the native picker selects a different handle.
                if (active !== this.selected && this.root.contains(document.activeElement) &&
                    document.activeElement.classList.contains('nrb-coordinate')) document.activeElement.blur();
                // Native click, Shift-click, box selection and history restoration
                // all enter here; none may include the protected endpoints.
                super.setSelection(indices.filter(i => this.options.editable.includes(i)), active);
            }
            syncEditingState() {
                const locked = !this.options.editable.includes(this.selected);
                for (const option of this.editing.pointSelect.options)
                    option.disabled = !this.options.editable.includes(Number(option.value));
                this.editing.coordinates.forEach((input, axis) => {
                    input.disabled ||= locked || axis === 2;
                });
                if (this.resetPoint) this.resetPoint.disabled = locked || isBusy(this);
            }
            updateSelection() {
                super.updateSelection();
                if (!this.editing || !this.model) return;
                const point = this.handles[this.selected]?.position;
                this.syncEditingState();
                this.editing.coordinates.forEach((input, axis) => {
                    if (point && axis < 2 && document.activeElement !== input) {
                        input.value = Number(point.getComponent(axis).toPrecision(9)).toString();
                        input.dataset.displayValue = input.value;
                    }
                });
                this.deltaOutputs?.forEach((output, axis) => {
                    if (!point) { output.value = '—'; return; }
                    const delta = point.getComponent(axis) -
                        session.blade.control_points[this.selected][axis];
                    output.value = Math.abs(delta) < 1e-12 ? '0' : Number(delta.toPrecision(7)).toString();
                });
                this.handles.forEach((handle, i) => {
                    if (!this.options.editable.includes(i) && this.fixedOutline)
                        handle.userData.outline.material = this.fixedOutline;
                });
            }
            updateGeometry(data) {
                if (Number.isFinite(data.pitch)) this.passagePitch = data.pitch;
                super.updateGeometry(data);
                // The periodic curves share the edited GPU buffer in both
                // preview and full quality. No separate spline evaluation.
                this.adjacentCurves?.forEach(line => { line.geometry = this.geometryObject.geometry; });
                this.adjacentGroup?.children.forEach(line => {
                    line.position.x = line.userData.periodicDirection * this.passagePitch;
                });
            }
        }
        const root = document.getElementById('cp-viewport');
        await new Promise(resolve => requestAnimationFrame(resolve));
        const evaluate = httpEvaluator(endpoint);
        const viewer = new StatorExplorer(root, session.scene, {
            editable: session.editable,
            evaluate: (packet, signal) => evaluate({session: {blade: session.blade, edits: session.edits,
                pitch_to_chord: session.pitch_to_chord}, packet}, signal),
            onBusyChange: publish,
        });
        state.viewer = viewer; current = state;
        const referenceLine = (points, options) => wideLine(lineGeometry(new Float32Array(points.flat())), options);
        viewer.baselineLine = referenceLine(session.references.baseline, {color: '#a5b2bb', width: 1.3, depthTest: false, order: -2});
        viewer.baselineLine.material.dashed = true; viewer.baselineLine.computeLineDistances();
        viewer.baselineLine.material.dashSize = .25; viewer.baselineLine.material.gapSize = .18;
        viewer.scene.add(viewer.baselineLine);
        const edge = referenceLine(session.references.trailing_edge, {color: '#233f50', width: 2.6, depthTest: false});
        viewer.scene.add(edge);
        viewer.adjacentGroup = new THREE.Group(); viewer.adjacentCurves = [];
        viewer.passagePitch = session.pitch;
        for (const direction of [-1, 1]) {
            const line = wideLine(viewer.geometryObject.geometry, {color: '#83949f', width: 1.7, depthTest: false, order: -1});
            const edgeLine = wideLine(edge.geometry, {color: '#83949f', width: 1.7, depthTest: false, order: -1});
            line.position.x = edgeLine.position.x = direction * session.pitch;
            line.userData.periodicDirection = edgeLine.userData.periodicDirection = direction;
            viewer.adjacentGroup.add(line, edgeLine); viewer.adjacentCurves.push(line);
        }
        viewer.scene.add(viewer.adjacentGroup);
        viewer.fixedOutline = viewer.outlineMaterial.clone(); viewer.fixedOutline.color.set('#a5b2bb');
        viewer.original = structuredClone(session.scene.model);
        viewer.original.control_points = [0, 1].map(axis => session.blade.control_points.map(p => p[axis]));
        viewer.original.control_points.push(session.blade.control_points.map(() => 0));
        const editCoordinate = viewer.editing.editCoordinate.bind(viewer.editing);
        viewer.editing.editCoordinate = (axis, raw) => {
            // Display fewer digits without turning an untouched field's Enter
            // or blur into a rounded geometry edit.
            if (axis < 2 && session.editable.includes(viewer.selected) &&
                raw !== viewer.editing.coordinates[axis].dataset.displayValue) editCoordinate(axis, raw);
        };
        const editing = viewer.editing;
        const beginChange = editing.beginChange.bind(editing);
        editing.beginChange = kind => {
            const started = beginChange(kind);
            if (started) state.notice = '';
            return started;
        };
        const updateHistory = editing.updateHistory.bind(editing);
        editing.updateHistory = () => {
            updateHistory();
            viewer.syncEditingState();
            publish();
        };
        viewer.viewModeSelect.disabled = true;
        configurePanel(viewer, session);
        viewer.options.bounds = () => {
            const box = new THREE.Box3().setFromPoints(viewer.handles.map(handle => handle.position));
            if (viewer.adjacentGroup.visible) { box.min.x -= viewer.passagePitch; box.max.x += viewer.passagePitch; }
            return box;
        };
        viewer.select(session.editable[0]); viewer.fit(); publish();
        return {mounted: true, identity: session.identity};
    } catch (error) {
        status('Editor unavailable: ' + error.message, true, true);
        return {error: error.message};
    }
}

export function capture() {
    if (!current) return {error: 'Open the control-point editor first.'};
    const e = current.viewer;
    if (isBusy(e)) return {error: 'Wait for the current edit to finish.'};
    if (e.displayQuality !== 'full') return {error: 'Wait for a fully evaluated blade before applying.'};
    return {identity: current.session.identity, model: structuredClone(e.acceptedModel), nonce: Date.now()};
}
