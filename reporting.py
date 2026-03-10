import base64
import os
from fpdf import FPDF

class PDFReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 16)
        self.cell(0, 10, 'Ofek: Maritime Event Impact Report', border=0, align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', border=0, align='C')

def create_impact_pdf_report(country, port, feature, event_date, resolution, 
                             total_actual, total_expected, absolute_diff, pct_diff, 
                             impact_text, exec_summary, img_path) -> bytes:
    
    pdf = PDFReport(orientation='P') # Portrait mode fits this single chart better
    pdf.add_page()
    
    # Header Info
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 10, 'Executive Summary', new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font('helvetica', '', 11)
    pdf.cell(0, 6, f"Location: {port}, {country}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Target Feature Analyzed: {feature} ({resolution} Resolution)", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Event Date: {event_date}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # Statistical Highlights
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, "Cumulative Impact Metrics:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('helvetica', '', 11)
    
    pdf.cell(0, 6, f"• Actual Post-Event: {total_actual:,.0f}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"• Expected (Counterfactual): {total_expected:,.0f}", new_x="LMARGIN", new_y="NEXT")
    
    # Color-code the net impact based on the text
    if "Positive" in impact_text: pdf.set_text_color(44, 160, 44)    # Green
    elif "Negative" in impact_text: pdf.set_text_color(214, 39, 40)  # Red
    else: pdf.set_text_color(127, 127, 127)                          # Gray
    
    pdf.cell(0, 6, f"• Net Impact: {absolute_diff:+,.0f} ({pct_diff:+.1f}%)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0) # Reset to black
    pdf.ln(8)
    
    # Insert Plotly Chart Image
    if os.path.exists(img_path):
        pdf.image(img_path, x=10, w=190)
        pdf.ln(5)
    
    # Plain English Summary
    pdf.set_font('helvetica', '', 11)
    pdf.multi_cell(0, 6, exec_summary.replace("**", "")) # Remove markdown formatting
    
    return bytes(pdf.output())
