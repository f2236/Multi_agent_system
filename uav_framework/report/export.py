import csv
from typing import List, Any


def export_positions_csv(path: str, uavs: List[Any], history: List[float]):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['uav_id', 'x', 'y', 'z'])
        for u in uavs:
            writer.writerow([getattr(u, 'uav_id', ''), u.pos[0], u.pos[1], u.pos[2]])

        # write a blank line then history
        writer.writerow([])
        writer.writerow(['iteration', 'objective'])
        for i, v in enumerate(history, start=1):
            writer.writerow([i, v])


def export_positions_pdf(path: str, uavs: List[Any], history: List[float]):
    """Create a simple PDF report of UAV positions and objective history.

    Falls back to a plain-text file if reportlab isn't available.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except Exception:
        # fallback: write a simple text file
        with open(path, 'w', newline='') as f:
            f.write('UAV Positions\n')
            for u in uavs:
                f.write(f"{getattr(u,'uav_id','')}, {u.pos[0]}, {u.pos[1]}, {u.pos[2]}\n")
            f.write('\nObjective History\n')
            for i, v in enumerate(history, start=1):
                f.write(f"{i}, {v}\n")
        return

    c = canvas.Canvas(path, pagesize=letter)
    w, h = letter
    y = h - 40
    c.setFont('Helvetica-Bold', 14)
    c.drawString(40, y, 'UAV Positions')
    y -= 20
    c.setFont('Helvetica', 10)
    for u in uavs:
        c.drawString(40, y, f"{getattr(u,'uav_id','')}: {u.pos[0]:.2f}, {u.pos[1]:.2f}, {u.pos[2]:.2f}")
        y -= 14
        if y < 80:
            c.showPage()
            y = h - 40

    if y < 140:
        c.showPage()
        y = h - 40

    c.setFont('Helvetica-Bold', 14)
    c.drawString(40, y, 'Objective History')
    y -= 20
    c.setFont('Helvetica', 10)
    for i, v in enumerate(history, start=1):
        c.drawString(40, y, f"{i}: {v}")
        y -= 12
        if y < 40:
            c.showPage()
            y = h - 40

    c.save()
