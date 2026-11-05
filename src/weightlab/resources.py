"""Sample host high-water memory and device VRAM without extra dependencies."""
from contextlib import contextmanager
from pathlib import Path
import threading


@contextmanager
def monitor(pid):
    metrics={'sample_interval_seconds':0.2,'host_vm_hwm_kib':0,'device_vram_used_bytes':0,
             'note':'Device-wide VRAM includes desktop/other users; sampled maximum may miss brief spikes.'}
    stop=threading.Event()
    devices=list(Path('/sys/class/drm').glob('card*/device/mem_info_vram_used'))
    def sample():
        while not stop.is_set():
            try:
                for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                    if line.startswith('VmHWM:'):metrics['host_vm_hwm_kib']=max(metrics['host_vm_hwm_kib'],int(line.split()[1]))
                for device in devices:metrics['device_vram_used_bytes']=max(metrics['device_vram_used_bytes'],int(device.read_text()))
            except (OSError,ValueError):pass
            stop.wait(.2)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    try:yield metrics
    finally:stop.set();thread.join(timeout=2)
