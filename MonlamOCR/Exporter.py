import abc
import logging
import numpy as np
import numpy.typing as npt
from xml.dom import minidom
import xml.etree.ElementTree as etree
from MonlamOCR.Data import BBox, Line, LayoutData, LineData
from MonlamOCR.Utils import get_utc_time, rotate_contour, optimize_countour


class Exporter:
    def __init__(self):
        logging.info("Init Exporter")

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'export_layout') and
                callable(subclass.export_layout) or
                NotImplemented)

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'export_lines') and
                callable(subclass.export_lines) or
                NotImplemented)

    @abc.abstractmethod
    def export_layout(self, image: npt.NDArray, image_name: str, layout_data: LayoutData, text_lines: list[str]):
        """Builds the characters et for encoding the labels."""
        raise NotImplementedError

    @abc.abstractmethod
    def export_lines(self, image: npt.NDArray, image_name: str, line_data: LineData, text_lines: list[str]):
        """Builds the characters et for encoding the labels."""
        raise NotImplementedError

    @staticmethod
    def get_bbox(bbox: BBox) -> tuple[int, int, int, int]:
        x = bbox.x
        y = bbox.y
        w = bbox.w
        h = bbox.h

        return x, y, w, h

    @staticmethod
    def get_text_points(contour):
        points = ""
        for box in contour:
            point = f"{box[0][0]},{box[0][1]} "
            points += point
        return points

    @staticmethod
    def get_bbox_points(bbox: tuple[int]):
        points = f"{bbox.x},{bbox.y} {bbox.x + bbox.w},{bbox.y} {bbox.x + bbox.w},{bbox.y + bbox.h} {bbox.x},{bbox.y + bbox.h}"
        return points
    

class PageXMLExporter(Exporter):
    def __init__(self) -> None:
        super().__init__()
        logging.info("Init XML Exporter")

    def get_text_line_block(self, coordinate, baseline_points, index, unicode_text):
        text_line = etree.Element(
            "Textline", id="", custom=f"readingOrder {{index:{index};}}"
        )
        text_line = etree.Element("TextLine")
        text_line_coords = coordinate

        text_line.attrib["id"] = f"line_9874_{str(index)}"
        text_line.attrib["custom"] = f"readingOrder {{index: {str(index)};}}"

        coords_points = etree.SubElement(text_line, "Coords")
        coords_points.attrib["points"] = text_line_coords
        baseline = etree.SubElement(text_line, "Baseline")
        baseline.attrib["points"] = baseline_points

        text_equiv = etree.SubElement(text_line, "TextEquiv")
        unicode_field = etree.SubElement(text_equiv, "Unicode")
        unicode_field.text = unicode_text

        return text_line

    def get_line_baseline(self, bbox: tuple[int, int, int, int]) -> str:
        return f"{bbox.x},{bbox.y + bbox.h} {bbox.x + bbox.w},{bbox.y + bbox.h}"

    def build_xml_document(self,
                           image: npt.NDArray,
                           image_name: str,
                           images: tuple[int],
                           text_bbox: BBox,
                           lines: list[Line],
                           margins: tuple[int],
                           captions: tuple[int],
                           text_lines: list[str] | None,
                           angle: float
                           ):
        root = etree.Element("PcGts")
        root.attrib[
            "xmlns"
        ] = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
        root.attrib["xmlns:xsi"] = "http://www.w3.org/2001/XMLSchema-instance"
        root.attrib[
            "xsi:schemaLocation"
        ] = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"

        metadata = etree.SubElement(root, "Metadata")
        creator = etree.SubElement(metadata, "Creator")
        creator.text = "Transkribus"
        created = etree.SubElement(metadata, "Created")
        created.text = get_utc_time()

        page = etree.SubElement(root, "Page")
        page.attrib["imageFilename"] = image_name
        page.attrib["imageWidth"] = f"{image.shape[1]}"
        page.attrib["imageHeight"] = f"{image.shape[0]}"

        reading_order = etree.SubElement(page, "ReadingOrder")
        ordered_group = etree.SubElement(reading_order, "OrderedGroup")
        ordered_group.attrib["id"] = f"1234_{0}"
        ordered_group.attrib["caption"] = "Regions reading order"

        region_ref_indexed = etree.SubElement(reading_order, "RegionRefIndexed")
        region_ref_indexed.attrib["index"] = "0"
        region_ref = "region_main"
        region_ref_indexed.attrib["regionRef"] = region_ref

        text_region = etree.SubElement(page, "TextRegion")
        text_region.attrib["id"] = region_ref
        text_region.attrib["custom"] = "readingOrder {index:0;}"

        text_region_coords = etree.SubElement(text_region, "Coords")
        text_region_coords.attrib["points"] = f"{text_bbox.x},{text_bbox.y} {text_bbox.x+text_bbox.w},{text_bbox.y} {text_bbox.x+text_bbox.w},{text_bbox.y+text_bbox.h} {text_bbox.x},{text_bbox.y+text_bbox.h}"

        do_rotate = False
        x_center = image.shape[1] // 2
        y_center = image.shape[0] // 2
        
        if angle != abs(0):
            do_rotate = True

        for i in range(0, len(lines)):
            #bbox_line_coords = self.get_bbox_points(lines[i].bbox)

            line_contour = lines[i].contour
            line_bbox = lines[i].bbox

            if do_rotate:
                line_contour = rotate_contour(line_contour, x_center, y_center, angle)
            
            line_contour = optimize_countour(line_contour)

            text_coords = self.get_text_points(line_contour)
            base_line_coords = self.get_line_baseline(line_bbox)

            if text_lines is not None and len(text_lines) > 0:
                text_region.append(
                    self.get_text_line_block(coordinate=text_coords, baseline_points=base_line_coords, index=i,
                                             unicode_text=text_lines[i])
                )
            else:
                text_region.append(
                    self.get_text_line_block(coordinate=text_coords, baseline_points=base_line_coords, index=i,
                                             unicode_text=""))

        if len(images) > 0:
            for idx, bbox in enumerate(images):
                image_region = etree.SubElement(page, "ImageRegion")
                image_region.attrib["id"] = "Image_1234"
                image_region.attrib["custom"] = f"readingOrder {{index: {str(idx)};}}"

                coords_points = etree.SubElement(image_region, "Coords")
                coords_points.attrib["points"] = self.get_bbox_points(bbox)

        if len(margins) > 0:
            for idx, bbox in enumerate(margins):
                margin_region = etree.SubElement(page, "TextRegion")
                margin_region.attrib["id"] = f"margin_1234_{idx}"
                margin_region.attrib["type"] = "margin"
                margin_region.attrib["custom"] = f"readingOrder {{index: {str(idx)};}} structure {{type:marginalia;}}"

                coords_points = etree.SubElement(margin_region, "Coords")
                coords_points.attrib["points"] = self.get_bbox_points(bbox)

        if len(captions) > 0:
            for idx, bbox in enumerate(captions):
                captions_region = etree.SubElement(page, "TextRegion")
                captions_region.attrib["id"] = f"caption_1234_{idx}"
                captions_region.attrib["type"] = "caption"
                captions_region.attrib["custom"] = f"readingOrder {{index: {str(idx)};}} structure {{type:caption;}}"

                coords_points = etree.SubElement(captions_region, "Coords")
                coords_points.attrib["points"] = self.get_bbox_points(bbox)

        parsed_xml = minidom.parseString(etree.tostring(root))
        parsed_xml = parsed_xml.toprettyxml()

        return parsed_xml

    def export_layout(self, image: np.array, image_name: str, layout_data: LayoutData, text_lines: list[str],
                      output_dir: str):
        image_boxes = [self.get_bbox(x) for x in layout_data.images]
        caption_boxes = [self.get_bbox(x) for x in layout_data.captions]
        margin_boxes = [self.get_bbox(x) for x in layout_data.margins]
        line_boxes = [self.get_bbox(x.bbox) for x in layout_data.lines]
        text_bbox = self.get_bbox(layout_data.text_bboxes[0])

        xml_doc = self.build_xml_document(
            image,
            image_name,
            images=image_boxes,
            lines=line_boxes,
            margins=margin_boxes,
            captions=caption_boxes,
            text_region_bbox=text_bbox,
            text_lines=text_lines
        )

        out_file = f"{output_dir}/{image_name}.xml"

        with open(out_file, "w", encoding='UTF-8') as f:
            f.write(xml_doc)

    def export_lines(self, image: np.array, image_name: str, line_data: LineData, text_lines: list[str],
                     output_dir: str):
        line_boxes = [self.get_bbox(x.bbox) for x in line_data.lines]
        text_bbox = self.get_bbox(line_data.bbox)

        xml_doc = self.build_xml_document(
            image,
            image_name,
            images=[],
            lines=line_boxes,
            margins=[],
            captions=[],
            text_region_bbox=text_bbox,
            text_lines=text_lines
        )

        out_file = f"{output_dir}/{image_name}.xml"

        with open(out_file, "w", encoding='UTF-8') as f:
            f.write(xml_doc)