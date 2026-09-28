"""Locally generated invented text/graphics/scanned PDFs; no real report data."""
from io import BytesIO
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, NumberObject


def research_pdf(kind="chart"):
    writer = PdfWriter()
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
    descriptions = [
        "Synthetic research introduction. Packaging is the report subject. This page supplies ordinary searchable text.",
        "Synthetic architecture diagram. The optical engine is adjacent to the invented switch.",
        "Synthetic table. Local package measurements are listed in the table.",
        "Synthetic CPO chart. The invented optical engine consumes 6 W at 20 C.",
    ]
    for index, description in enumerate(descriptions):
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        text = f"BT /F1 12 Tf 35 745 Td ({description}) Tj ET\n"
        if index == 1:
            text += "40 400 170 100 re S 350 400 170 100 re S 210 450 m 350 450 l S\n"
            text += "BT /F1 18 Tf 45 440 Td (Optical engine) Tj 310 0 Td (Switch) Tj ET"
        if index == 2:
            text += "40 300 500 200 re S 40 400 m 540 400 l S 280 300 m 280 500 l S\n"
            text += "BT /F1 18 Tf 60 450 Td (Device) Tj 250 0 Td (Power W) Tj -250 -100 Td (Engine) Tj 250 0 Td (6) Tj ET"
        if index == 3:
            text += "60 300 m 60 620 l S 60 300 m 540 300 l S 150 300 90 180 re S 360 300 90 270 re S\n"
            text += "BT /F1 18 Tf 65 650 Td (Power W) Tj 80 -175 Td (6 W) Tj 210 90 Td (9 W) Tj -205 -295 Td (20 C) Tj 200 0 Td (40 C) Tj ET"
        if kind == "conflict" and index == 3:
            text = text.replace("consumes 6 W", "consumes 10 W")
        if kind == "illegible" and index == 3:
            text = text.replace("/F1 18 Tf", "/F1 0.1 Tf")
        stream = DecodedStreamObject()
        stream.set_data(text.encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    if kind != "image_only":
        return output.getvalue()
    # Rasterize this invented chart into an image XObject: pypdf sees NO text.
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw
    with pdfium.PdfDocument(output.getvalue()) as document:
        page = document[3]
        try:
            bitmap = page.render(scale=1, force_bitmap_format=raw.FPDFBitmap_BGR, rev_byteorder=True)
            try:
                rgb = b"".join(bytes(bitmap.buffer[row * bitmap.stride:row * bitmap.stride + bitmap.width * 3])
                    for row in range(bitmap.height))
                width, height = bitmap.width, bitmap.height
            finally:
                bitmap.close()
        finally:
            page.close()
    scanned = PdfWriter()
    page = scanned.add_blank_page(width=612, height=792)
    image = DecodedStreamObject()
    image.set_data(rgb)
    image.update({NameObject("/Type"): NameObject("/XObject"), NameObject("/Subtype"): NameObject("/Image"),
        NameObject("/Width"): NumberObject(width), NameObject("/Height"): NumberObject(height),
        NameObject("/ColorSpace"): NameObject("/DeviceRGB"), NameObject("/BitsPerComponent"): NumberObject(8)})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/XObject"):
        DictionaryObject({NameObject("/Im0"): scanned._add_object(image.flate_encode())})})
    command = DecodedStreamObject()
    command.set_data(b"q 612 0 0 792 0 0 cm /Im0 Do Q")
    page[NameObject("/Contents")] = scanned._add_object(command)
    result = BytesIO()
    scanned.write(result)
    return result.getvalue()
