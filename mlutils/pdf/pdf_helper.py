import io

import fitz


class PDFHelper:
    """ Contains helper method for commonly used PDF extraction operations"""

    def __init__(self, file):
        if isinstance(file, io.IOBase):
            self.doc = fitz.open(stream=file, filetype="pdf")
        elif isinstance(file, str):
            self.doc = fitz.open(file, filetype="pdf")
        self.metadata = ()
        for page in self.doc:
            blocks = page.get_text('dict')['blocks']
            page_text = tuple(
                [(span['text'], span['bbox'], page.number) for block in blocks if not block['type'] for line in
                 block['lines'] for span in line['spans']])
            page_text = tuple(sorted(page_text, key=lambda x: x[1][1]))
            self.metadata = self.metadata + page_text
        self.converted_doc = fitz.open('pdf', self.doc.convert_to_pdf(from_page=0, to_page=self.doc.page_count))

    def get_images_by_page(self, page_no):
        """
        extracts all images from input page of a pdf document.
        Parameters:
            page_no <int>: The input PDF document Page number.
        Returns:
            <list>: The output list of images present on the input Page
        """
        return self.converted_doc[page_no].get_images(full=True)

    def find_page_by_text(self, text):
        """
        finds page number where input text is found
        Parameters:
            text <str>: The input text to be found.
        Returns:
            <list>: The list of page number(s) where the input text is found
        """
        output = list(filter(lambda x: text in x, self.metadata))
        return list(set([x[-1] for x in output])) if output else None

    def get_bbox_by_text(self, text, page_no=None):
        """
            Return Bounding Box and Page Number where is text is found on the page.
            If page number is not given it will return Bounding Box and Page Number
            for whole document where text is found.
            Parameters:
                text <str> : The input text to be found.
                page_no <int> : The input page_no of document.
            Return:
                <tuple> : The output bounding box and page number where text is found.
        """
        bbox = tuple(filter(lambda x: text == x[0].strip() and page_no == x[2],
                            self.metadata)) if page_no is not None else tuple(
            filter(lambda x: text == x[0].strip(), self.metadata))
        bbox = tuple(x[1:] for x in bbox)
        return bbox

    def get_attributes_by_page(self, page_no):
        """
            Return all attributes of input page of document.
            Parameters:
                page_no <int> : The input page_no of document.
            Return:
                <tuple> : The output text, bounding box and page number where page_no is found.
        """
        data = tuple(filter(lambda x: x[2] == page_no, self.metadata))
        return data

    def get_form_fields_by_page(self, page_no):
        """
            Return PDF Form field, also called a “widget”, as dictionary of field name as key and
            field value as dictionary value and returns None when no form field is found on the page.
            Parameters:
                page_no <int> : The input page_no of document.
            Return:
                fields <dict>: PDF Form field name and value for the given page no, as dictionary of field name
                as key and field value is dictionary value. {field_name: field_value}
        """
        fields = {field.field_name: field.field_value for field in self.doc[page_no].widgets()}
        return fields if fields else None
