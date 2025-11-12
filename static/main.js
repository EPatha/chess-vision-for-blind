document.addEventListener('DOMContentLoaded', () => {
  const startBtn = document.getElementById('start');
  const sourceInput = document.getElementById('source');
  const videoImg = document.getElementById('video');
  const snapshotBtn = document.getElementById('snapshotBtn');
  const calib = document.getElementById('calib');
  const snapCanvas = document.getElementById('snapCanvas');
  const sendPointsBtn = document.getElementById('sendPoints');
  const resetPointsBtn = document.getElementById('resetPoints');
  const pointsList = document.getElementById('pointsList');

  let points = [];

  startBtn.addEventListener('click', () => {
    const src = sourceInput.value.trim();
    if (!src) return alert('Provide source URL');
    videoImg.src = '/video_feed?source=' + encodeURIComponent(src);
  });

  snapshotBtn.addEventListener('click', async () => {
    const src = sourceInput.value.trim();
    if (!src) return alert('Provide source URL');
    const res = await fetch('/snapshot?source=' + encodeURIComponent(src));
    if (!res.ok) return alert('Snapshot failed: ' + res.statusText);
    const blob = await res.blob();
    const img = new Image();
    img.onload = () => {
      snapCanvas.width = img.width;
      snapCanvas.height = img.height;
      calib.style.display = 'block';
      const ctx = snapCanvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      points = [];
      pointsList.textContent = '';
    };
    img.src = URL.createObjectURL(blob);
  });

  snapCanvas.addEventListener('click', (ev) => {
    const rect = snapCanvas.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    const y = ev.clientY - rect.top;
    if (points.length >= 4) return alert('Already 4 points');
    points.push({x: Math.round(x), y: Math.round(y)});
    const ctx = snapCanvas.getContext('2d');
    ctx.fillStyle = 'rgba(255,0,0,0.9)';
    ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI*2); ctx.fill();
    ctx.font = '16px Arial'; ctx.fillText(String(points.length), x+8, y-8);
    pointsList.textContent = 'Points: ' + points.map(p => `(${p.x},${p.y})`).join(', ');
  });

  resetPointsBtn.addEventListener('click', () => {
    points = [];
    pointsList.textContent = '';
    // redraw image by reloading snapshot
    const evt = new Event('click');
    snapshotBtn.dispatchEvent(evt);
  });

  sendPointsBtn.addEventListener('click', async () => {
    if (points.length !== 4) return alert('Click exactly 4 points in TL,TR,BR,BL order');
    const res = await fetch('/set_corners', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({points})
    });
    const j = await res.json();
    if (!j.ok) return alert('set_corners failed: ' + (j.error || JSON.stringify(j)));
    alert('Corners set — now the video feed will show mapped squares');
    calib.style.display = 'none';
  });
});
