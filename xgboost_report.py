from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
import os

# Create PDF
pdf_file = 'model_report/xgboost_performance_report.pdf'
if not os.path.exists('model_report'):
    os.makedirs('model_report')

doc = SimpleDocTemplate(pdf_file, pagesize=letter, rightMargin=0.5*inch, leftMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
styles = getSampleStyleSheet()
title_style = ParagraphStyle(name='Title', fontSize=18, leading=22, alignment=1, spaceAfter=20)
heading_style = ParagraphStyle(name='Heading2', fontSize=14, leading=16, spaceAfter=12)
body_style = ParagraphStyle(name='Body', fontSize=11, leading=13, spaceAfter=10)

elements = []

# Title Page
elements.append(Paragraph("Passenger Flow Prediction Report", title_style))
elements.append(Paragraph("AI Model Performance Summary", styles['Heading2']))
elements.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y %I:%M %p EAT')}", body_style))
elements.append(Paragraph("Prepared for General Review", body_style))
elements.append(Spacer(1, 0.5*inch))

# Section 1: Overview
elements.append(Paragraph("1. Purpose of This Report", heading_style))
elements.append(Paragraph(
    "This report evaluates an AI model designed to predict passenger numbers for public transport. "
    "The model uses 500,000 trip records, analyzing factors such as time of day, weather, and weekends. "
    "It predicts exact passenger counts (e.g., 44 or 80 passengers) and categorizes them as Low (0–50), Medium (51–100), or High (over 100). "
    "Training took 2.5 seconds, and predictions are made in 0.004 seconds, enabling real-time applications.",
    body_style
))
elements.append(Spacer(1, 0.25*inch))

# Section 2: Prediction Accuracy
elements.append(Paragraph("2. Prediction Accuracy", heading_style))
elements.append(Paragraph(
    "The model was tested on 100,000 trips. The following table summarizes its performance, including the R² score, which shows how well the model explains changes in passenger numbers:",
    body_style
))
table_data = [
    ["Measure", "Result", "Explanation"],
    ["Average Error (MAE)", "18 passengers", "Predictions are typically off by about 18 passengers"],
    ["R² Score", "60% (0.6034)", "The model explains 60% of passenger number variations"],
    ["Root Mean Squared Error (RMSE)", "21.73 passengers", "Larger errors are around 22 passengers"],
    ["Consistency", "High", "Performance is stable across different data sets"],
]
table = Table(table_data)
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
]))
elements.append(table)
elements.append(Spacer(1, 0.25*inch))

# Section 3: Factors Influencing Passenger Numbers
elements.append(Paragraph("3. Factors Influencing Passenger Numbers", heading_style))
elements.append(Paragraph(
    "The chart below identifies the key factors affecting passenger numbers. "
    "<b>Key Insights</b>: Busy hours (rush times) have the largest impact (72.6%), followed by weekends (15.7%) and weather (8.8%). "
    "This suggests that peak travel times and weekend patterns are critical for predicting passenger counts, while specific dates have less influence.",
    body_style
))
if os.path.exists('model_report/feature_importance.png'):
    elements.append(Image('model_report/feature_importance.png', width=5*inch, height=3*inch))
else:
    elements.append(Paragraph("Feature importance chart not found.", body_style))
elements.append(Spacer(1, 0.25*inch))

# Section 4: Accuracy of Predictions
elements.append(Paragraph("4. Accuracy of Predictions", heading_style))
elements.append(Paragraph(
    "The chart below shows how close predictions are to actual passenger numbers. "
    "<b>Key Insights</b>: Most predictions are accurate (within 18 passengers), but larger errors occur during unusual times, such as holidays or bad weather, indicating a need for better data in these scenarios.",
    body_style
))
if os.path.exists('model_report/error_distribution.png'):
    elements.append(Image('model_report/error_distribution.png', width=5*inch, height=3*inch))
else:
    elements.append(Paragraph("Error distribution chart not found.", body_style))
elements.append(Spacer(1, 0.25*inch))

# Section 5: Passenger Group Classification
elements.append(Paragraph("5. Passenger Group Classification", heading_style))
elements.append(Paragraph(
    "The model categorizes passenger counts as Low (0–50), Medium (51–100), or High (over 100), with an overall accuracy of 67%. "
    "<b>Key Insights</b>: Predictions are reliable for Medium and High groups (busy and typical times) but less accurate for Low groups (quiet times), often predicting quiet periods as busier. "
    "This could lead to scheduling more buses than needed during quiet times.",
    body_style
))
table_data = [
    ["Group", "Accuracy", "Explanation"],
    ["Low (0–50)", "Low (21%)", "Quiet times are often predicted as busier"],
    ["Medium (51–100)", "High (80%)", "Very accurate for typical trips"],
    ["High (>100)", "Good (67%)", "Reliable for busy times, with some errors"],
]
table = Table(table_data)
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
]))
elements.append(table)
if os.path.exists('model_report/confusion_matrix.png'):
    elements.append(Image('model_report/confusion_matrix.png', width=4*inch, height=3*inch))
else:
    elements.append(Paragraph("Passenger group classification chart not found.", body_style))
elements.append(Spacer(1, 0.25*inch))

# Section 6: Example Predictions
elements.append(Paragraph("6. Example Predictions", heading_style))
elements.append(Paragraph(
    "The following examples illustrate the model’s predictions:",
    body_style
))
elements.append(Paragraph("- Wednesday, 8:00 AM, Sunny, Non-Busy: 44 passengers", body_style))
elements.append(Paragraph("- Friday, 5:00 PM, Rainy, Busy: 80 passengers", body_style))
elements.append(Paragraph("- Sunday, 12:00 PM, Sunny, Weekend: 59 passengers", body_style))
elements.append(Spacer(1, 0.25*inch))

# Section 7: Strengths and Areas for Improvement
elements.append(Paragraph("7. Strengths and Areas for Improvement", heading_style))
elements.append(Paragraph(
    "<b>Strengths:</b>",
    body_style
))
elements.append(Paragraph(
    "- <b>Fast Predictions</b>: Predictions are generated in 0.004 seconds, enabling real-time scheduling and resource allocation.",
    body_style
))
elements.append(Paragraph(
    "- <b>Reliable for Busy and Typical Times</b>: The model accurately predicts passenger numbers during peak hours (Medium and High groups, with 80% and 67% accuracy, respectively), ensuring effective planning for busy periods.",
    body_style
))
elements.append(Paragraph(
    "- <b>Consistent Performance</b>: The model maintains stable accuracy across different data sets (train RMSE: 21.71, test RMSE: 21.76), indicating robust generalization.",
    body_style
))
elements.append(Paragraph(
    "<b>Areas for Improvement:</b>",
    body_style
))
elements.append(Paragraph(
    "- <b>Poor Accuracy for Quiet Times</b>: The model struggles to predict low passenger counts (Low group, 21% accuracy), often misclassifying quiet periods as busier, which may lead to over-scheduling buses.",
    body_style
))
elements.append(Paragraph(
    "- <b>Limited Overall Accuracy</b>: The R² score of 60% (0.6034) indicates that the model explains only 60% of passenger number variations, leaving room for improvement in capturing all trends.",
    body_style
))
elements.append(Paragraph(
    "- <b>Limited Data for Unusual Conditions</b>: Larger errors occur during holidays or bad weather, suggesting that the current data lacks sufficient information for these scenarios.",
    body_style
))
elements.append(Spacer(1, 0.25*inch))

# Section 8: Future Improvements
elements.append(Paragraph("8. Future Improvements", heading_style))
elements.append(Paragraph(
    "The following steps can enhance the model’s performance:",
    body_style
))
elements.append(Paragraph(
    "<b>Improved Data Collection:</b>",
    body_style
))
elements.append(Paragraph(
    "- <b>Gather Real-World Data</b>: Collect ticket sales, bus GPS data, or station passenger counts to reflect actual travel patterns, particularly during holidays or events.",
    body_style
))
elements.append(Paragraph(
    "- <b>Incorporate Additional Factors</b>: Include data on special events (e.g., concerts), road closures, or public holidays to better predict unusual travel patterns.",
    body_style
))
elements.append(Paragraph(
    "- <b>Ensure Data Accuracy</b>: Verify data quality by cross-checking multiple sources and correcting errors, such as missing or incorrect times.",
    body_style
))
elements.append(Paragraph(
    "<b>Advanced Computing:</b>",
    body_style
))
elements.append(Paragraph(
    "- <b>Utilize NVIDIA GPUs</b>: Train the model on a computer with NVIDIA GPUs (e.g., RTX 3080 or A100) using XGBoost’s GPU support (set tree_method='hist', device='cuda') to process larger datasets and optimize settings faster.",
    body_style
))
elements.append(Paragraph(
    "- <b>Optimize Model Settings</b>: Test various model configurations (e.g., number of trees, learning rate) through a systematic grid search to improve accuracy.",
    body_style
))
elements.append(Paragraph(
    "<b>Enhanced Model Intelligence:</b>",
    body_style
))
elements.append(Paragraph(
    "- <b>Add New Factors</b>: Incorporate time-of-day patterns (e.g., morning vs. evening) or traffic data to capture additional travel trends.",
    body_style
))
elements.append(Paragraph(
    "- <b>Improve Quiet Time Predictions</b>: Apply techniques like SMOTE to balance data for quiet periods, enhancing predictions for low passenger counts.",
    body_style
))
elements.append(Paragraph(
    "- <b>Explore Advanced Models</b>: Experiment with neural networks or combined models (e.g., XGBoost with LightGBM) to potentially achieve better performance.",
    body_style
))

# Build PDF
doc.build(elements)
print(f"PDF report saved as {pdf_file}")