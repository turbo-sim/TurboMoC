/* Sidebar width is a browser preference, independent of scientific inputs. */
(() => {
  const storageKey = "turbomoc.sidebarWidth";
  let mounted, cleanup;

  function mount() {
    const divider = document.getElementById("sidebar-divider");
    const sidebar = document.getElementById("sidebar");
    if (!divider || !sidebar || mounted === divider) return;
    if (cleanup) cleanup();
    mounted = divider;
    let preferred = sidebar.getBoundingClientRect().width;
    try {
      const saved = Number(localStorage.getItem(storageKey));
      if (Number.isFinite(saved) && saved > 0) preferred = saved;
    } catch (_) { /* Resizing also works when browser storage is disabled. */ }
    let drag = null, frame = null;

    function bounds() {
      const style = getComputedStyle(sidebar);
      const min = parseFloat(style.minWidth) || 250;
      const max = Math.max(min, Math.min(parseFloat(style.maxWidth) || 600,
        sidebar.parentElement.clientWidth - divider.offsetWidth - 320));
      return {min, max};
    }

    function notify() {
      if (frame !== null) return;
      frame = requestAnimationFrame(() => {
        frame = null;
        // Plotly listens for window resizing; VTK observes its own container.
        window.dispatchEvent(new Event("resize"));
      });
    }

    function apply(width, remember = false) {
      const {min, max} = bounds();
      const value = Math.round(Math.max(min, Math.min(max, width)));
      if (sidebar.style.width !== value + "px") {
        sidebar.style.width = value + "px";
        notify();
      }
      divider.setAttribute("aria-valuemin", Math.round(min));
      divider.setAttribute("aria-valuemax", Math.round(max));
      divider.setAttribute("aria-valuenow", value);
      if (remember) {
        preferred = value;
        try { localStorage.setItem(storageKey, value); } catch (_) {}
      }
    }

    function finish(event) {
      if (!drag || (event && event.pointerId !== drag.id)) return;
      const id = drag.id;
      drag = null;
      document.body.classList.remove("sidebar-resizing");
      if (divider.hasPointerCapture(id)) divider.releasePointerCapture(id);
      apply(sidebar.getBoundingClientRect().width, true);
    }

    divider.addEventListener("pointerdown", event => {
      if (event.button !== 0 || drag) return;
      event.preventDefault();
      drag = {id: event.pointerId, x: event.clientX, width: sidebar.getBoundingClientRect().width};
      divider.setPointerCapture(event.pointerId);
      divider.focus();
      document.body.classList.add("sidebar-resizing");
    });
    divider.addEventListener("pointermove", event => {
      if (drag && event.pointerId === drag.id) apply(drag.width + event.clientX - drag.x);
    });
    divider.addEventListener("pointerup", finish);
    divider.addEventListener("pointercancel", finish);
    divider.addEventListener("lostpointercapture", finish);
    divider.addEventListener("keydown", event => {
      const {min, max} = bounds();
      const current = sidebar.getBoundingClientRect().width;
      const width = {ArrowLeft: current - 16, ArrowRight: current + 16, Home: min, End: max}[event.key];
      if (width === undefined) return;
      event.preventDefault();
      apply(width, true);
    });
    divider.addEventListener("dblclick", () => apply(380, true));
    const resize = () => { if (!drag) apply(preferred); };
    window.addEventListener("resize", resize);
    apply(preferred);
    cleanup = () => {
      finish();
      window.removeEventListener("resize", resize);
      if (frame !== null) cancelAnimationFrame(frame);
    };
  }

  new MutationObserver(mount).observe(document.documentElement, {childList: true, subtree: true});
  mount();
})();
