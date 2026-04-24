from io import BytesIO
from PIL import Image
from PyQt6.QtGui import QPixmap


def pil_to_qpixmap(image: Image.Image) -> QPixmap:
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    pixmap = QPixmap()
    pixmap.loadFromData(buffer.getvalue())
    return pixmap
