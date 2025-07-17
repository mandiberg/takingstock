import pandas as pd
import os
import math
from reportlab.pdfgen import canvas
from PyPDF2 import PdfReader, PdfWriter
import tempfile

#using reportlab to create and style the pdf
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer, PageTemplate, Frame, BaseDocTemplate, PageBreak, Flowable
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import Color

#need pdfmetrics to register the font
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

#global vars
input_path = "input_csvs/"

#CSV Vars
inputCSV = 'input_csvs/topic_13_counts.csv'
number_of_topics = 37*2 #37 lines per page, 2 pages per spread, one spread per csv
csv_list = {}

#Color analysis vars
LOG_COLOR = True
LOG_VAL = 50
LOG_SUB = 1 #rough need to estimate
BLACK_COUNT = 10000

# div count by 1000, if count_color > 1 count_color = 1
# PDF Vars
outputPDF = 'outputs/list/output.pdf'
page_size = [432, 648] #in points, 72 points = 1 inch
#margin settings (in points) (72 points = 1 inch)
INNER_MARGIN = 90  # 1.25 inch
OUTER_MARGIN = 54  # 0.75 inch
TOP_MARGIN = 54
BOTTOM_MARGIN = 54
font_size = 12
font_file = 'fonts/CrimsonText-Regular.ttf'
footer_font_file = 'fonts/CrimsonText-SemiBold.ttf'
footer_text = "TOPIC: "  # Base text, topic number will be added dynamically

# take the top number_of_topics and put them in a new dataframe
# then sort them by description alphabetically
# then save that section of data into a dictionary where the csv number is the key and the value is the table

def analyze_csv(input_csv, input_path, num_rows):
    dict_key = [int(s) for s in input_csv.split('_') if s.isdigit()]
    dict_key = str(dict_key)[1:len(dict_key)-2]
    df = pd.read_csv(input_path+input_csv)
    df = df.dropna()
    df = df.head(num_rows)
    df = df.sort_values(by="description", ascending = True)
    return dict_key, len(df), df

#go through the input path and add dataframes to list
for file in sorted(os.listdir(input_path)):
    if file.endswith(".csv"):
        csv_info = analyze_csv(file, input_path, number_of_topics)
        csv_list.update({csv_info[0] : csv_info[2]})  # Use topic number (csv_info[0]) as key, not length

# PDF Work Starts
class BlankPage(Flowable):
    def __init__(self):
        Flowable.__init__(self)
        self.width = 0
        self.height = 528

    def draw(self):
        pass

# Set column width for single column - will be adjusted per page
column_width = [page_size[0] - INNER_MARGIN - OUTER_MARGIN]  # Full width minus margins
# Convert the data to paragraphs for proper wrapping
pdf_elements = [BlankPage()]  # Start with blank page

# Register the custom font
pdfmetrics.registerFont(TTFont('CrimsonText', font_file))
pdfmetrics.registerFont(TTFont('CrimsonText-SemiBold', footer_font_file))


# Create style for the table
styles = getSampleStyleSheet()
table_style = ParagraphStyle(
    'TableStyle',
    parent=styles['Normal'],
    fontSize=font_size,
    leading=14,
    spaceBefore=0,
    spaceAfter=0,
    fontName='CrimsonText' 
)

# Create style for the footer text
footer_style = ParagraphStyle(
    'FooterStyle',
    parent=styles['Normal'],
    fontSize=font_size,
    leading=14,
    spaceBefore=0,
    spaceAfter=0,
    fontName='CrimsonText-SemiBold',
    alignment=1  # Center alignment
)


def generate_two_page_topic_pdf(topic_key, topic_data, table_style, col_width, output_path):

    def count_to_color(this_count):
        if LOG_COLOR: 
            gray_val = math.log(this_count, LOG_VAL)
            gray_val = gray_val - LOG_SUB #this is rough I want to better estimate this
        else:
            gray_val = max(0.0, min(1.0, this_count / BLACK_COUNT))
        print('gray_val: ' + str(gray_val))
        print('count:' + str(this_count))
        if gray_val > 1:
            gray_val = 1
        gray_val = 1 - gray_val
        gray_color = Color(gray_val, gray_val, gray_val)
        return gray_color

    # Try adding rows until we get exactly 2 pages
    max_rows = len(topic_data)
    min_rows = 1
    best_rows = 1
    # Binary search for the max number of rows that fit in 2 pages
    while min_rows <= max_rows:
        mid = (min_rows + max_rows) // 2
        # Build the PDF in memory
        tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        tmp_pdf.close()
        doc = BaseDocTemplate(tmp_pdf.name, pagesize=page_size,
                             topMargin=TOP_MARGIN, bottomMargin=BOTTOM_MARGIN,
                             leftMargin=0, rightMargin=0)
        frame = Frame(OUTER_MARGIN, BOTTOM_MARGIN, page_size[0] - OUTER_MARGIN - INNER_MARGIN, page_size[1] - TOP_MARGIN - BOTTOM_MARGIN, id='normal')
        template = PageTemplate(id='normal', frames=[frame])
        doc.addPageTemplates([template])
        rows = []
        for _, row in topic_data.head(mid).iterrows():
            para = Paragraph(str(row.iloc[1]), table_style)
            rows.append([para])
        table = Table(rows, colWidths=col_width)
        table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('FONTNAME', (0,0), (-1,-1), 'CrimsonText'), 
        ]))
        doc.build([table])
        reader = PdfReader(tmp_pdf.name)
        num_pages = len(reader.pages)
        os.unlink(tmp_pdf.name)
        if num_pages == 2:
            best_rows = mid
            min_rows = mid + 1
        elif num_pages < 2:
            min_rows = mid + 1
        else:
            max_rows = mid - 1
    
    # Now generate the final 2-page PDF with best_rows and footer
    doc = BaseDocTemplate(output_path, pagesize=page_size,
                         topMargin=TOP_MARGIN, bottomMargin=BOTTOM_MARGIN,
                         leftMargin=0, rightMargin=0)
    
    # Create frames for odd and even pages with alternating margins
    odd_frame = Frame(OUTER_MARGIN, BOTTOM_MARGIN + 1, 
                     page_size[0] - OUTER_MARGIN - INNER_MARGIN, 
                     page_size[1] - TOP_MARGIN - BOTTOM_MARGIN, id='odd')
    even_frame = Frame(INNER_MARGIN, BOTTOM_MARGIN + 1, 
                      page_size[0] - OUTER_MARGIN - INNER_MARGIN, 
                      page_size[1] - TOP_MARGIN - BOTTOM_MARGIN, id='even')
    
    # Create page templates with footer function
    def add_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('CrimsonText-SemiBold', font_size)
        current_footer_text = footer_text + str(topic_key)
        
        # Choose footer alignment based on even/odd page
        is_even = doc.page % 2 == 0
        x = (page_size[0] - (OUTER_MARGIN * 2)) if is_even else OUTER_MARGIN
        
        canvas.drawString(x, BOTTOM_MARGIN - 20, current_footer_text)
        canvas.restoreState()
    
    odd_template = PageTemplate(id='odd', frames=[odd_frame], onPage=add_footer)
    even_template = PageTemplate(id='even', frames=[even_frame], onPage=add_footer)
    
    doc.addPageTemplates([odd_template, even_template])
    doc.pageTemplate = odd_template  # Set initial template
    
    rows = []
    for _, row in topic_data.head(best_rows).iterrows():
        gray_color = count_to_color(row.iloc[0])

                # Create a paragraph style with dynamic grayscale color
        dynamic_style = ParagraphStyle(
            'DynamicStyle',
            parent=table_style,
            textColor=gray_color,
        )

        para = Paragraph(str(row.iloc[1]), dynamic_style)
        rows.append([para])
    table = Table(rows, colWidths=col_width)
    table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('FONTNAME', (0,0), (-1,-1), 'CrimsonText'), 
    ]))
    doc.build([table])

def create_blank_page_pdf(output_path):
    """Create a blank page PDF"""
    doc = BaseDocTemplate(output_path, pagesize=page_size,
                         topMargin=TOP_MARGIN, bottomMargin=BOTTOM_MARGIN,
                         leftMargin=0, rightMargin=0)
    frame = Frame(OUTER_MARGIN, BOTTOM_MARGIN, page_size[0] - OUTER_MARGIN - INNER_MARGIN, page_size[1] - TOP_MARGIN - BOTTOM_MARGIN, id='normal')
    template = PageTemplate(id='normal', frames=[frame])
    doc.addPageTemplates([template])
    # Create an empty flowable for the blank page
    blank_flowable = BlankPage()
    doc.build([blank_flowable])

# Process all topics in csv_list
temp_pdf_files = []

# Create blank page first
print("Creating blank page...")
blank_pdf_path = "temp_blank_page.pdf"
create_blank_page_pdf(blank_pdf_path)
temp_pdf_files.append(blank_pdf_path)

for topic_key, topic_data in csv_list.items():
    print(f"Processing topic {topic_key}...")
    temp_pdf_path = f"temp_topic_{topic_key}.pdf"
    generate_two_page_topic_pdf(topic_key, topic_data, table_style, column_width, temp_pdf_path)
    temp_pdf_files.append(temp_pdf_path)

print("Merging PDFs...")
# Merge all temporary PDFs
merger = PdfWriter()
for pdf_file in temp_pdf_files:
    with open(pdf_file, 'rb') as file:
        merger.append(file)
    os.remove(pdf_file)  # Clean up temporary file

# Write the merged PDF
with open(outputPDF, 'wb') as output_file:
    merger.write(output_file)

print(f"PDF saved to {outputPDF}")
