# ============================================================
#   HYPERTENSION PREDICTION - FLASK APP WITH PDF REPORT
#   Run: python app.py
#   Open: http://127.0.0.1:5000
# ============================================================

from flask import Flask, render_template, request, flash, make_response
import joblib
import numpy as np
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io

app = Flask(__name__)
app.secret_key = 'hypertension_secret_key'

# Load model
try:
    model = joblib.load('logreg_model.pkl')
    print("✓ Model loaded!")
except:
    model = None
    print("✗ Model not found!")

stage_map  = {0:'NORMAL', 1:'HYPERTENSION (Stage-1)', 2:'HYPERTENSION (Stage-2)', 3:'HYPERTENSIVE CRISIS'}
color_map  = {0:'#10B981', 1:'#F59E0B', 2:'#F97316', 3:'#EF4444'}
risk_map   = {0:'LOW RISK', 1:'MODERATE RISK', 2:'HIGH RISK', 3:'EMERGENCY'}

recommendations = {
    0: {
        'title': 'Normal Blood Pressure',
        'description': 'Great news! Your cardiovascular risk assessment indicates normal blood pressure levels.',
        'actions': [
            'Maintain your current healthy lifestyle',
            'Exercise at least 150 minutes per week',
            'Continue a balanced low-sodium diet',
            'Get blood pressure checked annually',
            'Avoid smoking and limit alcohol',
            'Manage stress through yoga or meditation'
        ],
        'lifestyle_tips': [
            {'icon':'🏃','title':'Stay Active','detail':'30 mins of moderate exercise daily reduces risk by 35%'},
            {'icon':'🥗','title':'Eat Smart','detail':'DASH diet: rich in fruits, vegetables and low-fat dairy'},
            {'icon':'😴','title':'Sleep Well','detail':'7-8 hours of quality sleep keeps blood pressure stable'},
            {'icon':'🧘','title':'Manage Stress','detail':'Chronic stress raises BP — try 10 min daily meditation'}
        ]
    },
    1: {
        'title': 'Stage 1 Hypertension',
        'description': 'Mild elevation detected. Lifestyle changes now can prevent progression to Stage 2.',
        'actions': [
            'Schedule a doctor appointment within 2 weeks',
            'Start the DASH diet immediately',
            'Reduce sodium to less than 1,500mg/day',
            'Exercise 30 minutes, 5 days a week',
            'Monitor blood pressure every 2-3 days',
            'Quit smoking and reduce alcohol'
        ],
        'lifestyle_tips': [
            {'icon':'🧂','title':'Cut Salt','detail':'Reducing sodium by 1,000mg/day lowers BP by 5-6 mmHg'},
            {'icon':'🏊','title':'Aerobic Exercise','detail':'Swimming or brisk walking 5x/week helps significantly'},
            {'icon':'🍌','title':'Potassium Foods','detail':'Bananas, spinach, avocado help counteract sodium'},
            {'icon':'📱','title':'Monitor Daily','detail':'Track BP morning and evening, share logs with doctor'}
        ]
    },
    2: {
        'title': 'Stage 2 Hypertension',
        'description': 'Significant hypertension detected. Immediate medical intervention required.',
        'actions': [
            'URGENT: See a doctor within 1-2 days',
            'Medication therapy is very likely required',
            'Get a comprehensive cardiovascular assessment',
            'Monitor blood pressure twice daily',
            'Strictly limit sodium to 1,000mg/day',
            'Avoid strenuous activity until cleared by doctor'
        ],
        'lifestyle_tips': [
            {'icon':'💊','title':'Medication','detail':'ACE inhibitors may be prescribed — take consistently'},
            {'icon':'🚫','title':'Avoid Triggers','detail':'Caffeine, alcohol and stress can spike BP dangerously'},
            {'icon':'📋','title':'Keep Records','detail':'Log every BP reading with time — essential for doctor'},
            {'icon':'👨‍👩‍👧','title':'Family Support','detail':'Inform family — they should know emergency procedures'}
        ]
    },
    3: {
        'title': 'Hypertensive Crisis',
        'description': 'CRITICAL: Dangerously elevated blood pressure. This is a medical emergency.',
        'actions': [
            'EMERGENCY: Seek immediate medical attention NOW',
            'Call emergency services (108/112) immediately',
            'Do NOT drive yourself — call an ambulance',
            'Sit down, stay calm, avoid physical exertion',
            'Monitor for chest pain, vision loss, confusion',
            'Go to the nearest emergency department NOW'
        ],
        'lifestyle_tips': [
            {'icon':'🚨','title':'Call Emergency','detail':'India emergency: 108 | 112. Act immediately'},
            {'icon':'🛑','title':'Stop Everything','detail':'Sit or lie down, breathe slowly, avoid movement'},
            {'icon':'📞','title':'Alert Someone','detail':'Tell someone nearby your condition immediately'},
            {'icon':'🏥','title':'ER Immediately','detail':'Find the nearest emergency room right now'}
        ]
    }
}


def encode_inputs(form):
    gender          = 0 if form['Gender'] == 'Male' else 1
    age             = {'18-34':1,'35-50':2,'51-64':3,'65+':4}[form['Age']]
    history         = 1 if form['History'] == 'Yes' else 0
    patient         = 1 if form['Patient'] == 'Yes' else 0
    take_medication = 1 if form['TakeMedication'] == 'Yes' else 0
    severity        = {'Mild':0,'Moderate':1,'Severe':2}[form['Severity']]
    breath          = 1 if form['BreathShortness'] == 'Yes' else 0
    visual          = 1 if form['VisualChanges'] == 'Yes' else 0
    nose            = 1 if form['NoseBleeding'] == 'Yes' else 0
    when_diagnosed  = {'<1 Year':1,'1 - 5 Years':2,'>5 Years':3}[form['Whendiagnoused']]
    systolic        = {'100 - 110':0,'111 - 120':1,'121 - 130':2,'130+':3}[form['Systolic']]
    diastolic       = {'70 - 80':0,'81 - 90':1,'91 - 100':2,'100+':3}[form['Diastolic']]
    diet            = 1 if form['ControlledDiet'] == 'Yes' else 0

    input_array = np.array([[
        gender,
        (age-1)/3,
        history, patient, take_medication,
        severity/2,
        breath, visual, nose,
        (when_diagnosed-1)/2,
        systolic/3,
        diastolic/3,
        diet
    ]])
    return input_array


# ── PDF GENERATOR ─────────────────────────────────────────────
def generate_pdf(form_data, stage_name, risk_level, confidence, color_hex, rec):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    # Color from hex
    def hex_to_color(h):
        h = h.lstrip('#')
        return colors.Color(*[int(h[i:i+2],16)/255 for i in (0,2,4)])

    stage_color = hex_to_color(color_hex)
    navy = colors.Color(0.07, 0.13, 0.25)
    light_gray = colors.Color(0.95, 0.96, 0.97)

    styles = getSampleStyleSheet()
    story  = []

    # ── HEADER ──
    title_style = ParagraphStyle('title', fontName='Helvetica-Bold',
                                  fontSize=22, textColor=navy,
                                  alignment=TA_CENTER, spaceAfter=4)
    sub_style = ParagraphStyle('sub', fontName='Helvetica',
                                fontSize=10, textColor=colors.gray,
                                alignment=TA_CENTER, spaceAfter=2)

    story.append(Paragraph("❤️  PulsePred", title_style))
    story.append(Paragraph("AI-Powered Hypertension Risk Assessment Report", sub_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", sub_style))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=stage_color))
    story.append(Spacer(1, 0.4*cm))

    # ── RESULT BOX ──
    result_data = [[
        Paragraph(f"<b>{stage_name}</b>", ParagraphStyle('r1', fontName='Helvetica-Bold', fontSize=14, textColor=colors.white)),
        Paragraph(f"<b>{risk_level}</b>", ParagraphStyle('r2', fontName='Helvetica-Bold', fontSize=12, textColor=colors.white, alignment=TA_CENTER)),
        Paragraph(f"<b>Confidence: {confidence}%</b>", ParagraphStyle('r3', fontName='Helvetica-Bold', fontSize=12, textColor=colors.white, alignment=TA_CENTER)),
    ]]
    result_table = Table(result_data, colWidths=[9*cm, 4*cm, 4*cm])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), stage_color),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [stage_color]),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 10),
        ('ROUNDEDCORNERS', [8,8,8,8]),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 0.3*cm))

    # Description
    desc_style = ParagraphStyle('desc', fontName='Helvetica-Oblique',
                                 fontSize=10, textColor=colors.gray,
                                 alignment=TA_CENTER, spaceAfter=12)
    story.append(Paragraph(rec['description'], desc_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Spacer(1, 0.3*cm))

    # ── PATIENT DETAILS ──
    sec_style = ParagraphStyle('sec', fontName='Helvetica-Bold',
                                fontSize=12, textColor=navy, spaceBefore=8, spaceAfter=6)
    story.append(Paragraph("👤  Patient Details", sec_style))

    detail_items = [
        ['Gender', form_data.get('Gender','—')],
        ['Age Group', form_data.get('Age','—')],
        ['Family History', form_data.get('History','—')],
        ['Under Medical Care', form_data.get('Patient','—')],
        ['Taking Medication', form_data.get('TakeMedication','—')],
        ['Symptom Severity', form_data.get('Severity','—')],
        ['Shortness of Breath', form_data.get('BreathShortness','—')],
        ['Vision Changes', form_data.get('VisualChanges','—')],
        ['Nosebleeds', form_data.get('NoseBleeding','—')],
        ['Since Diagnosed', form_data.get('Whendiagnoused','—')],
        ['Systolic BP', form_data.get('Systolic','—')],
        ['Diastolic BP', form_data.get('Diastolic','—')],
        ['Controlled Diet', form_data.get('ControlledDiet','—')],
    ]

    label_s = ParagraphStyle('lbl', fontName='Helvetica-Bold', fontSize=9, textColor=navy)
    val_s   = ParagraphStyle('val', fontName='Helvetica', fontSize=9, textColor=colors.black)

    # Split into 2 columns
    half = len(detail_items)//2 + len(detail_items)%2
    left_col  = detail_items[:half]
    right_col = detail_items[half:]

    rows = []
    for i in range(half):
        l = left_col[i]
        r = right_col[i] if i < len(right_col) else ['','']
        rows.append([
            Paragraph(l[0], label_s), Paragraph(l[1], val_s),
            Paragraph('', val_s),
            Paragraph(r[0], label_s), Paragraph(r[1], val_s),
        ])

    details_table = Table(rows, colWidths=[3.5*cm, 3.5*cm, 0.5*cm, 3.5*cm, 3.5*cm])
    details_table.setStyle(TableStyle([
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [light_gray, colors.white]),
        ('PADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (1,-1), 0.5, colors.lightgrey),
        ('GRID', (3,0), (4,-1), 0.5, colors.lightgrey),
    ]))
    story.append(details_table)
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))

    # ── RECOMMENDATIONS ──
    story.append(Paragraph("📋  Clinical Recommendations", sec_style))
    for i, action in enumerate(rec['actions'], 1):
        item_style = ParagraphStyle(f'item{i}', fontName='Helvetica',
                                     fontSize=10, textColor=colors.black,
                                     leftIndent=10, spaceAfter=4)
        bullet_style = ParagraphStyle(f'bul{i}', fontName='Helvetica-Bold',
                                       fontSize=10, textColor=stage_color)
        row = Table([[Paragraph(f"{i}.", bullet_style), Paragraph(action, item_style)]],
                    colWidths=[0.7*cm, 16.3*cm])
        row.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('PADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(row)

    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))

    # ── LIFESTYLE TIPS ──
    story.append(Paragraph("💡  Lifestyle Tips", sec_style))
    tip_rows = []
    tips = rec['lifestyle_tips']
    for i in range(0, len(tips), 2):
        left_tip  = tips[i]
        right_tip = tips[i+1] if i+1 < len(tips) else None
        left_cell  = Paragraph(f"<b>{left_tip['icon']} {left_tip['title']}</b><br/>{left_tip['detail']}",
                                ParagraphStyle('tip', fontName='Helvetica', fontSize=9, textColor=colors.black))
        right_cell = Paragraph(f"<b>{right_tip['icon']} {right_tip['title']}</b><br/>{right_tip['detail']}",
                                ParagraphStyle('tip2', fontName='Helvetica', fontSize=9, textColor=colors.black)) if right_tip else Paragraph('', styles['Normal'])
        tip_rows.append([left_cell, right_cell])

    tips_table = Table(tip_rows, colWidths=[8.5*cm, 8.5*cm])
    tips_table.setStyle(TableStyle([
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [light_gray, colors.white]),
        ('PADDING', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(tips_table)
    story.append(Spacer(1, 0.5*cm))

    # ── FOOTER ──
    story.append(HRFlowable(width="100%", thickness=2, color=stage_color))
    footer_style = ParagraphStyle('footer', fontName='Helvetica-Oblique',
                                   fontSize=8, textColor=colors.gray,
                                   alignment=TA_CENTER, spaceBefore=6)
    story.append(Paragraph(
        "⚠️ This report is generated by an AI model for educational purposes only. "
        "It is NOT a substitute for professional medical advice. "
        "Always consult a qualified healthcare provider for diagnosis and treatment.",
        footer_style
    ))
    story.append(Paragraph("PulsePred — Built with Python, Flask & scikit-learn", footer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ── ROUTES ────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        form = request.form
        input_array = encode_inputs(form)
        prediction  = model.predict(input_array)[0]
        confidence  = round(max(model.predict_proba(input_array)[0]) * 100, 1)

        return render_template('index.html',
            prediction_text = stage_map[prediction],
            result_color    = color_map[prediction],
            result_risk     = risk_map[prediction],
            confidence      = confidence,
            recommendation  = recommendations[prediction],
            form_data       = form)
    except Exception as e:
        flash(f"Error: {str(e)}", "error")
        return render_template('index.html')


@app.route('/download_report', methods=['POST'])
def download_report():
    try:
        form = request.form
        input_array = encode_inputs(form)
        prediction  = model.predict(input_array)[0]
        confidence  = round(max(model.predict_proba(input_array)[0]) * 100, 1)

        pdf_buffer = generate_pdf(
            form_data  = form,
            stage_name = stage_map[prediction],
            risk_level = risk_map[prediction],
            confidence = confidence,
            color_hex  = color_map[prediction],
            rec        = recommendations[prediction]
        )

        response = make_response(pdf_buffer.read())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = \
            f'attachment; filename=HypertensionReport_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        return response

    except Exception as e:
        flash(f"PDF Error: {str(e)}", "error")
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
