document.getElementById('year')?.append(new Date().getFullYear());

document.querySelectorAll('.code-block').forEach(block => {
  const btn = document.createElement('button');
  btn.textContent = 'Copy';
  btn.className = 'btn ghost';
  btn.style.float = 'right'; btn.style.margin = '0 0 8px 8px';
  const pre = block.closest('pre') || block;
  pre.parentNode.insertBefore(btn, pre);
  btn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(block.innerText);
      btn.textContent = 'Copied!'; setTimeout(() => btn.textContent = 'Copy', 1200);
    } catch { btn.textContent = 'Failed'; setTimeout(() => btn.textContent = 'Copy', 1200); }
  });
});
