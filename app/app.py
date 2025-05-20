import os
import sys
import fitz
from quart import Quart, render_template, request, send_from_directory, current_app
from werkzeug.utils import secure_filename
import logging

from model.model_studybook import TextAnalyzer

app = Quart(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

app.secret_key = os.getenv('SECRET_KEY')

UPLOAD_FOLDER = 'temp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

#* Функция для извлечения текста из PDF
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

@app.route('/', methods=['GET', 'POST'])
async def index():
    if request.method == 'POST':
        file = await request.files 

        if 'file' not in file:
            return await render_template("index.html", error='Ошибка: нет файла')

        uploaded_file = file['file']

        if uploaded_file.filename == '':
            return await render_template("index.html", error='Ошибка: файл не выбран')

        if uploaded_file and uploaded_file.filename.endswith('.pdf'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(uploaded_file.filename))
            await uploaded_file.save(file_path) 
            try:
                text = extract_text_from_pdf(file_path)
                analyzer = TextAnalyzer()

                style_result = analyzer.classify_style(text)
                coherence_result, coherence_interpretation = await analyzer.analyze_coherence(text)
                original_text = extract_text_from_pdf('C:\\Programming\\Project_BERT\\data\\PDF_учебники\\1_УчП_Эконометрика_РД_Воскобойников.pdf')
                structure_result, structure_interpret = analyzer.analyze_structure(original_text, text)    
                read_result = await analyzer.analyze_readability(text)
                keywords = await analyzer.extract_keywords(text)

                if not read_result:
                    logging.error("Ошибка анализа")
                    read_result = "Не удалось проанализировать читаемость"
                
            except Exception as e:
                logging.error(f"Ошибка анализа: {str(e)}")
                return await render_template("index.html", error=f'Ошибка при анализе: {str(e)}')

            return await render_template("index.html",
                                        style_result=style_result,
                                        coherence_result=float(f"{coherence_result:.2f}"),
                                        coherence_interpretation=coherence_interpretation,
                                        structure_result=float(f'{structure_result:.2f}'),
                                        structure_interpret=structure_interpret,
                                        read_result=read_result,
                                        keywords=keywords)

    return await render_template("index.html")

@app.route('/favicon.ico')
async def favicon():
    return await send_from_directory(
        os.path.join(current_app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)

