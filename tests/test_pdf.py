import pickle

from mlutils.pdf.digital_pdf_helper import PDFHelper

pdf_obj = PDFHelper('./data/pdf/ALLIANCE_APP_Alex Vigil.pdf')
form_field_pdf_obj = PDFHelper('./data/pdf/01ArtisanBilingual(11_19_2021).pdf')


class TestPDFHelper:
    def test_get_images_by_page_valid_1(self):
        total_images = len(pdf_obj.get_images_by_page(page_no=4))
        assert total_images == 2

    def test_get_images_by_page_valid_2(self):
        total_images = len(pdf_obj.get_images_by_page(page_no=0))
        assert total_images != 2

    def test_get_images_by_page_invalid(self):
        try:
            x = len(pdf_obj.get_images_by_page(page_no=14))
        except IndexError:
            assert True

    def test_find_page_by_text_valid_1(self):
        page_numbers = pdf_obj.find_page_by_text(text='07/10/2021')
        assert page_numbers == [4, 5, 7, 8, 10, 11]

    def test_find_page_by_text_valid_2(self):
        page_numbers = pdf_obj.find_page_by_text(text='ALEX')
        assert page_numbers is None

    def test_get_bbox_by_text_valid_1(self):
        bbox_page_number = pdf_obj.get_bbox_by_text(text='ALEX VIGIL', page_no=0)
        assert bbox_page_number == (((92.000732421875, 165.1520233154297, 141.9416961669922, 174.1520233154297), 0),)

    def test_get_bbox_by_text_valid_2(self):
        bbox_page_number = pdf_obj.get_bbox_by_text(text='MIL4970933')
        result = (((92.00071716308594, 147.53591918945312, 159.34471130371094, 159.53591918945312), 0),
                  ((45.5, 152.45213317871094, 96.00798797607422, 161.45213317871094), 1),
                  ((87.5, 152.67210388183594, 138.0079803466797, 161.67210388183594), 11),
                  ((28.455900192260742, 127.35716247558594, 72.91990661621094, 135.35716247558594), 12))
        assert bbox_page_number == result

    def test_get_bbox_by_text_valid_3(self):
        bbox_page_number = pdf_obj.get_bbox_by_text(text='ALEX VIGILI')
        assert bbox_page_number == ()

    def test_get_attributes_by_page_valid_1(self):
        attributes = pdf_obj.get_attributes_by_page(page_no=12)
        with open('data/pdf/alex_vigil_tuple.pickle', 'rb') as f:
            data = pickle.load(f)
        assert attributes == data

    def test_get_attributes_by_page_valid_2(self):
        attributes = pdf_obj.get_attributes_by_page(page_no=13)
        assert attributes == ()

    def test_get_form_fields_by_page(self):
        fields = form_field_pdf_obj.get_form_fields_by_page(page_no=0)
        with open('./data/pdf/field_dictionary.pickle', 'rb') as f:
            data = pickle.load(f)
        assert fields == data

    def test_get_form_fields_by_page_invalid(self):
        try:
            fields = form_field_pdf_obj.get_form_fields_by_page(page_no=1)
        except IndexError:
            assert True