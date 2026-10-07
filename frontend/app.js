// Real-Time Industrial Dashboard Telemetry Controller

const COLOR_MAP = {
  "RED": "#ef4444",
  "BLUE": "#3b82f6",
  "YELLOW": "#eab308",
  "BLACK": "#9ca3af",
  "EMPTY": "#64748b",
  "NONE": "#475569"
};

async function updateTelemetry() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) return;
    const data = await res.json();

    // 1. Header Badges
    const camBadge = document.getElementById('camera-badge');
    if (camBadge) camBadge.textContent = `Camera Device: ${data.device_index ?? 1}`;
    
    const fpsBadge = document.getElementById('fps-badge');
    if (fpsBadge) fpsBadge.textContent = `${data.fps ?? 0.0} FPS`;
    
    const frameCounter = document.getElementById('frame-counter');
    if (frameCounter) frameCounter.textContent = `Frames: ${data.frame_count ?? 0}`;

    const select = document.getElementById('cam-select');
    if (select && data.device_index !== undefined && select.value != data.device_index) {
      select.value = data.device_index;
    }

    // 2. PASS / FAIL / ASSEMBLING Status Banner
    const banner = document.getElementById('status-banner');
    const statusTitle = document.getElementById('status-title');
    const statusReason = document.getElementById('status-reason');

    if (banner && statusTitle && statusReason) {
      const status = data.status || "SEARCHING";
      banner.className = "status-banner";

      if (status === "PASS") {
        banner.classList.add("banner-pass");
        statusTitle.textContent = "VERIFIED: PASS";
      } else if (status === "FAIL") {
        banner.classList.add("banner-fail");
        statusTitle.textContent = "DEFECT: REJECTED";
      } else if (status === "ASSEMBLING") {
        banner.classList.add("banner-idle");
        statusTitle.textContent = "ASSEMBLING";
      } else {
        banner.classList.add("banner-idle");
        statusTitle.textContent = "SEARCHING";
      }

      statusReason.textContent = data.diagnostic || "System running inspection";
    }

    // 3. Update TB1 - TB6 Matrix Cards
    const terminals = data.terminals || {};
    for (let i = 1; i <= 6; i++) {
      const tbKey = `TB${i}`;
      const tbInfo = terminals[tbKey];
      const tbElem = document.getElementById(`tb-${i}`);
      if (!tbElem || !tbInfo) continue;

      const expColor = tbInfo.expected || "EMPTY";
      const detColor = tbInfo.detected || "NONE";
      const isMatch = tbInfo.match;

      const expElem = tbElem.querySelector('.tb-exp');
      if (expElem) {
        expElem.textContent = `EXP: ${expColor}`;
        expElem.style.color = COLOR_MAP[expColor] || "#94a3b8";
      }

      const actElem = tbElem.querySelector('.tb-act');
      if (actElem) {
        const matchTag = isMatch ? "✓" : "✗";
        actElem.textContent = `DET: ${detColor} ${matchTag}`;
        actElem.style.color = isMatch ? "#22c55e" : "#ef4444";
      }

      // Border highlight based on inspection status
      if (isMatch) {
        tbElem.style.borderColor = "rgba(34, 197, 94, 0.4)";
        tbElem.style.backgroundColor = "rgba(34, 197, 94, 0.05)";
      } else {
        tbElem.style.borderColor = "rgba(239, 68, 68, 0.4)";
        tbElem.style.backgroundColor = "rgba(239, 68, 68, 0.05)";
      }
    }

    // 4. Production Metrics
    const metrics = data.metrics || {};
    const totalElem = document.getElementById('metric-total');
    if (totalElem) totalElem.textContent = metrics.total_cycles || 0;

    const passElem = document.getElementById('metric-pass');
    if (passElem) passElem.textContent = metrics.passed_cycles || 0;

    const failElem = document.getElementById('metric-fail');
    if (failElem) failElem.textContent = metrics.failed_cycles || 0;

    const cycleElem = document.getElementById('metric-cycle');
    if (cycleElem) cycleElem.textContent = `${metrics.cycle_time_ms || 0} ms`;

  } catch (err) {
    console.error("Telemetry fetch error:", err);
  }
}

async function switchCamera(deviceIndex) {
  try {
    const res = await fetch(`/api/switch_camera/${deviceIndex}`, { method: 'POST' });
    const data = await res.json();
    console.log("Switched camera:", data);
    const streamImg = document.getElementById('live-stream');
    if (streamImg) {
      streamImg.src = `/video_feed?t=${Date.now()}`;
    }
  } catch (err) {
    console.error("Error switching camera:", err);
  }
}

// Refresh telemetry every 300ms for smooth live updates
setInterval(updateTelemetry, 300);
updateTelemetry();
