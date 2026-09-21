from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle

OUT = Path('output/pdf/MedVision_AI_Team_Progress_Pitch.pdf')

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='DeckTitle', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=28, leading=34, textColor=colors.HexColor('#0A2858'), alignment=TA_CENTER, spaceAfter=18))
styles.add(ParagraphStyle(name='DeckSubtitle', parent=styles['BodyText'], fontSize=13, leading=20, alignment=TA_CENTER, textColor=colors.HexColor('#53708F')))
styles.add(ParagraphStyle(name='Section', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=22, leading=28, textColor=colors.HexColor('#0A2858'), spaceAfter=12))
styles.add(ParagraphStyle(name='Sub', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=13, leading=17, textColor=colors.HexColor('#0B5CAD'), spaceBefore=10, spaceAfter=5))
styles.add(ParagraphStyle(name='Talk', parent=styles['BodyText'], fontSize=11, leading=16, leftIndent=10, borderColor=colors.HexColor('#BBD8F4'), borderWidth=1, borderPadding=11, backColor=colors.HexColor('#F4F8FC'), spaceAfter=10))
styles.add(ParagraphStyle(name='Body2', parent=styles['BodyText'], fontSize=10.5, leading=15, spaceAfter=6))

def P(text, style='Body2'): return Paragraph(text, styles[style])
def page_number(canvas, doc):
    canvas.saveState(); canvas.setFont('Helvetica', 8); canvas.setFillColor(colors.HexColor('#637188'))
    canvas.drawString(42, 24, 'MedVision AI - Team Progress Presentation')
    canvas.drawRightString(A4[0] - 42, 24, f'Page {doc.page}'); canvas.restoreState()
   
members = [
('Person 1 - Frontend', '01_Member_One_Frontend', [
 ('Open first', 'Open <b>index.html</b> in a browser. Start at the Dashboard and use the left navigation.'),
 ('Show these items', 'Dashboard cards, recent activity, image upload flow, upload progress, experimental result card, case history search/filter, case-detail popup, and settings.'),
 ('Say this', '"I worked on the user-facing MedVision AI workflow. I started by building a responsive clinical dashboard so a reviewer can immediately see activity, pending work, and recent cases. Then I created the image-upload screen with PNG/JPEG validation, preview, size limits, and progress feedback. The interface does not treat the result as a diagnosis - every output is clearly labelled experimental."'),
 ('Continue with', 'Click <b>Case history</b>, search for MV-1024, change a status filter, then click View. Mention the reusable CSV export helper in <b>case-export.js</b>.'),
 ('Close with', '"My next task is to connect these screens to the live API and replace prototype data with case data returned by the backend."')]),
('Person 2 - Backend', '02_Member_Two_Backend', [
 ('Open first', 'Open <b>main.py</b>, then <b>test_api.py</b> and <b>test_case_service.py</b>.'),
 ('Show these items', 'Health check, accepted image metadata, allowed types and size limit, analysis job creation, status lookup, temporary completion callback, and automated API-flow tests.'),
 ('Say this', '"I built the first backend service layer. The API validates incoming image metadata before it creates an image record. Once the record exists, it creates an analysis job and lets the frontend track the job status. I also added a temporary completion route so the frontend can test the full workflow before the model worker is connected."'),
 ('Continue with', 'Show <b>case_service.py</b>: owner-scoped listing, pagination validation, and prevention of duplicate reference codes.'),
 ('Close with', '"My next step is to replace the in-memory prototype storage with the PostgreSQL schema and connect analysis completion to the AI worker."')]),
('Person 3 - AI Pipeline', '03_Member_Three_AI_Pipeline', [
 ('Open first', 'Open <b>preprocess.py</b>, then <b>batch_report.py</b>.'),
 ('Show these items', 'Supported-format checks, minimum image dimensions, contrast screening, orientation correction, resize to 224 x 224, normalization, and channel-first model tensor output.'),
 ('Say this', '"I prepared the input pipeline for the first baseline model. Before an image reaches a model, the pipeline checks that it is usable, fixes orientation, checks basic quality, resizes it to a consistent size, and normalizes the pixel values. This makes experiments reproducible and avoids feeding clearly bad images to the model."'),
 ('Continue with', 'Open <b>dataset_split.py</b> to show patient-level train, validation, and test splitting. Then show <b>metrics.py</b> for sensitivity, specificity, precision, and F1.'),
 ('Close with', '"My next task is to run the approved sample dataset through this pipeline, train a baseline, and document the results with clinical review."')]),
('Person 4 - Data and Security', '04_Member_Four_Data_Security', [
 ('Open first', 'Open <b>schema.sql</b> and scroll through the table definitions.'),
 ('Show these items', 'Users, cases, medical images, analysis jobs, results, and audit events. Point out file type and size constraints, the case-owner row-level policy, and audit trigger.'),
 ('Say this', '"I designed the first data and safety foundation. The schema separates users, cases, images, jobs, and audit events so the system can track an analysis from upload to result. I added database-level image restrictions, case-owner access policy, and audit logging for important case changes."'),
 ('Continue with', 'Open <b>002_integrity_and_reporting.sql</b>: completed jobs require results and completion time, indexes improve common queries, and the case-analysis view supports a future dashboard.'),
 ('Close with', '"My next task is to apply this to a local development database, create the least-privilege application role, and complete retention and storage-encryption settings."')])]

story = [Spacer(1, 1.4*inch), P('MedVision AI', 'DeckTitle'), P('Team Progress Update - Speaker-ready walkthrough', 'DeckSubtitle'), Spacer(1, .4*inch), P('This document is arranged so each person can open their own folder, demonstrate their work, and present the corresponding speaking notes.', 'DeckSubtitle'), PageBreak(), P('How to present this update', 'Section'), P('<b>Suggested order:</b> Person 1 demonstrates the product experience, Person 2 explains the workflow behind it, Person 3 shows how images are prepared for modelling, and Person 4 closes with data safety and system readiness.'), P('<b>Important wording:</b> Describe all output as experimental and for review. Do not say that the system diagnoses medical conditions.'), P('<b>Presentation flow for each person:</b> open the listed file, show the items in order, read the dialogue naturally, and finish with the next step. This creates a clear story from visible product experience to implementation foundation.'), PageBreak()]

for idx, (name, folder, items) in enumerate(members, 1):
    story += [P(f'{idx}. {name}', 'Section'), P(f'<b>Folder to open:</b> {folder}'), P('What this section proves', 'Sub')]
    story.append(P('This workstream contains a concrete, reviewable module that is ready to be connected with the other team modules in the next implementation phase.'))
    for label, text in items:
        story.append(P(label, 'Sub'))
        story.append(P(text, 'Talk' if label == 'Say this' else 'Body2'))
    story.append(PageBreak())

story += [P('Combined project status', 'Section'), P('The team has a working prototype foundation across interface, service workflow, model preparation, and data controls. The modules are deliberately separated at this stage so each part can be reviewed, tested, and improved before full integration.'), P('What has been demonstrated', 'Sub'), Table([[P('Area'), P('Current evidence')], [P('Frontend'), P('Interactive dashboard, upload journey, result state, case review')], [P('Backend'), P('Validated API prototype, analysis lifecycle, unit-test coverage')], [P('AI pipeline'), P('Preprocessing, quality reporting, split strategy, metrics')], [P('Data and security'), P('Schema, ownership controls, audit trail, integrity rules')]], colWidths=[1.55*inch, 5.35*inch], style=TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#0A2858')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#D8E2EF')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8)])), Spacer(1, .25*inch), P('Next integration phase', 'Sub'), P('Connect the frontend to the API, attach persistent PostgreSQL storage, run approved sample images through the pipeline, and review baseline model results before any wider use.'), P('Thank you', 'DeckTitle')]

OUT.parent.mkdir(parents=True, exist_ok=True)
SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=42, leftMargin=42, topMargin=48, bottomMargin=42).build(story, onFirstPage=page_number, onLaterPages=page_number)
print(OUT)
