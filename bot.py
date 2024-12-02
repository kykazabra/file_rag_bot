from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import os
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from PyPDF2 import PdfReader
import telebot
import traceback
from copy import deepcopy
import shutil
from telebot import types
import docx
from dotenv import load_dotenv
import json

load_dotenv()

llm = ChatOpenAI(
    openai_api_key=os.environ['OPENAI_API_KEY'],
    model_name='gpt-3.5-turbo',
    base_url='https://api.proxyapi.ru/openai/v1',
    temperature=0,
    verbose=False
)

embeddings = OpenAIEmbeddings(
    openai_api_key=os.environ['OPENAI_API_KEY'],
    base_url='https://api.proxyapi.ru/openai/v1',
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

col_dir = Path('collections')

n_retrieve = 3

os.makedirs(col_dir, exist_ok=True)

base_prompt = ChatPromptTemplate.from_messages([
    ("human", """Ты - помощник для ответа на вопросы. Используй куски полученного контекста ниже чтобы ответить на вопрос. Если ты не знаешь, что ответить, просто ответь, что не знаешь.
Вопрос: {question} 
Контекст: {context} 
Ответ:"""),
])


def pdf_to_text(path):
    reader = PdfReader(path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + '\n' * 2

    return text


def docx_to_text(path):
    doc = docx.Document(path)
    fullText = []

    for para in doc.paragraphs:
        fullText.append(para.text)

    return '\n\n'.join(fullText)


def txt_to_text(path):
    with open(path, 'r') as f:
        return f.read()


def add_collection(text, name):
    docs = text_splitter.create_documents([text])

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(col_dir / Path(name)),
    )

    return True


def load_collection(vcs_name):
    vcs = Chroma(persist_directory=str(col_dir / Path(vcs_name)),
                 embedding_function=embeddings)

    return vcs


def get_chain(vcs_name):
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    vectorstore = load_collection(vcs_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": n_retrieve})

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | base_prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain


def ask_question(vsc_name, question):
    chain = get_chain(vsc_name)

    ans = chain.invoke(question)

    del chain

    return ans


def inbi():
    if not 'bot_info.json' in os.listdir('.'):
        with open('bot_info.json', 'w') as f:
            json.dump({'users': [], 'admins': [], 'user_collections': {}, 'collections': []}, f)


inbi()


def with_user_validation(func):
    def wrapper(message):
        if message.from_user.username in users:
            return func(message)
        else:
            return None

    return wrapper


def with_admin_validation(func):
    def wrapper(message):
        if message.from_user.username in admins:
            return func(message)
        else:
            return None

    return wrapper


def update_bot_info():
    with open('bot_info.json', 'w') as f:
        json.dump({'users': users, 'admins': admins, 'user_collections': user_collections, 'collections': collections},
                  f)


def bot_command(func):
    def wrapper(message):
        try:
            cmd, user = message.text.strip().split()
            bot.reply_to(message, func(user))
            update_bot_info()
        except:
            bot.reply_to(message, f'Не могу выполнить операцию, команда должна быть в формате "/команда ник"')

    return wrapper


bot = telebot.TeleBot(token=os.environ['TG_BOT_API_KEY'])


@bot.message_handler(commands=['add_user'])
@with_admin_validation
@bot_command
def add_user(name):
    global users
    global user_collections

    if name not in users:
        users.append(name)
        user_collections[name] = None
        return f'Пользователь {name} добавлен'
    else:
        return f'Пользователь {name} уже в списке'


@bot.message_handler(commands=['drop_user'])
@with_admin_validation
@bot_command
def drop_user(name):
    global users

    if name in users:
        users.remove(name)
        return f'Пользователь {name} удален'
    else:
        return f'Пользователь {name} итак не в списке'


@bot.message_handler(commands=['add_admin'])
@with_admin_validation
@bot_command
def add_admin(name):
    global admins
    global users

    if name not in admins:
        admins.append(name)

        if name not in users:
            users.append(name)

        return f'Админ {name} добавлен'
    else:
        return f'Админ {name} уже в списке'


@bot.message_handler(commands=['drop_admin'])
@with_admin_validation
@bot_command
def drop_admin(name):
    global admins

    if name in admins:
        admins.remove(name)
        return f'Админ {name} удален'
    else:
        return f'Админ {name} итак не в списке'


@bot.message_handler(commands=['get_users'])
@with_user_validation
def bot_get_users(message):
    bot.reply_to(message, f"Список пользователей:\n{users}")


@bot.message_handler(commands=['get_admins'])
@with_admin_validation
def bot_get_admins(message):
    bot.reply_to(message, f"Список адмиинов:\n{admins}")


@bot.message_handler(commands=['start'])
@with_user_validation
def bot_start(message):
    bot.send_message(message.from_user.id, """
Привет, я помогу тебе с поиском информации!

Чтобы загрузить в меня документ, отправь мне команду /add_file <ключевое слово> (твое имя для файла).
После этого я буду готов отвечать на вопросы по твоему файлу.

Если захочешь сменить файл, по которому я должен отвечать, введи /set_file <ключевое слово>.

Если хочешь посмотреть список имен уже загруженных файлов, введи /view_files.
""")


user_should_name = {}
user_should_load = {}


@bot.message_handler(commands=['add_file'])
@with_user_validation
def bot_add_file(message):
    global user_should_load

    user_should_name[message.from_user.username] = True

    bot.reply_to(message, 'Как назовем файл?')

    update_bot_info()


@bot.message_handler(commands=['set_file'])
@with_user_validation
def bot_set_file(message):
    markup_inline = types.InlineKeyboardMarkup()

    for file in collections:
        markup_inline.add(types.InlineKeyboardButton(text=file, callback_data=f'set.{file}'))

    bot.reply_to(message, 'Выбреите файл', reply_markup=markup_inline)


@bot.callback_query_handler(
    func=lambda call: call.data.split('.')[1] in collections and call.data.split('.')[0] == 'set')
@with_user_validation
def bot_handle_set_file(call):
    global user_collections

    cmd, file = call.data.split('.')

    message = call.message

    user_collections[call.from_user.username] = file

    bot.send_message(call.from_user.id, f'Теперь я буду отвечать по файлу {file}!')

    bot.delete_message(call.from_user.id, message.id)

    update_bot_info()


@bot.message_handler(commands=['view_files'])
@with_user_validation
def bot_view_files(message):
    c = '\n'.join([f'*{co}*' for co in collections])

    bot.send_message(message.from_user.id, f'Все активные файлы:\n{c}', parse_mode='MARKDOWN')


@bot.message_handler(commands=['drop_file'])
@with_admin_validation
def bot_drop_file(message):
    global collections

    markup_inline = types.InlineKeyboardMarkup()

    for file in collections:
        markup_inline.add(types.InlineKeyboardButton(text=file, callback_data=f'drop.{file}'))

    bot.reply_to(message, 'Какой файл хотите удалить?', reply_markup=markup_inline)


@bot.callback_query_handler(
    func=lambda call: call.data.split('.')[1] in collections and call.data.split('.')[0] == 'drop')
@with_admin_validation
def bot_handle_drop_file(call):
    global collections
    global user_collections

    cmd, file = call.data.split('.')

    message = call.message

    collections.remove(file)

    uc = deepcopy(user_collections)
    for u, c in uc.items():
        if c == file:
            user_collections[u] = None

    shutil.rmtree(col_dir / file, ignore_errors=True)

    bot.send_message(call.from_user.id, f'Файл {file} успешно удален!')

    update_bot_info()

    bot.delete_message(call.from_user.id, message.id)


@bot.message_handler(content_types=['document'])
@with_user_validation
def bot_load_document(message):
    global collections
    global user_collections
    global user_should_load

    alias = user_should_load.get(message.from_user.username)

    if not alias:
        return None

    fmts = ['pdf', 'docx', 'txt']
    ok_flag = False
    fname = message.document.file_name

    for fmt in fmts:
        if fname.endswith(f'.{fmt}'):
            ok_flag = True

    if not ok_flag:
        bot.reply_to(message, 'Сори, с таким форматом не работаем (')
        return None

    bot.reply_to(message, f'Загружаю файл с кодовым именем *{alias}*', parse_mode='MARKDOWN')

    try:
        file_info = bot.get_file(message.document.file_id)

        if fname.endswith('.pdf'):
            lname = 'input.pdf'

            with open(lname, "wb") as f:
                file_content = bot.download_file(file_info.file_path)
                f.write(file_content)

            text = pdf_to_text(lname)

        elif fname.endswith('.docx'):
            lname = 'input.docx'

            with open(lname, "wb") as f:
                file_content = bot.download_file(file_info.file_path)
                f.write(file_content)

            text = docx_to_text(lname)

        elif fname.endswith('.txt'):
            lname = 'input.txt'

            with open(lname, "wb") as f:
                file_content = bot.download_file(file_info.file_path)
                f.write(file_content)

            text = txt_to_text(lname)

        if not text.strip():
            bot.reply_to(message, 'Я не смог достать информацию из твоего файла (')
            return None

        add_collection(text, alias)

        collections.append(alias)
        user_collections[message.from_user.username] = alias
        user_should_load[message.from_user.username] = None

        update_bot_info()

        bot.reply_to(message, f'Файл с кодовым именем {alias} загружен!')

    except:
        bot.reply_to(message, traceback.format_exc())


@bot.message_handler(content_types=['text'])
@with_user_validation
def bot_handle_text(message):
    global user_should_load
    global user_should_name

    if user_should_name.get(message.from_user.username):
        if message.text in collections:
            bot.reply_to(message, 'Файл с таким именем уже есть, напиши другое')
            return None

        user_should_name[message.from_user.username] = False
        user_should_load[message.from_user.username] = message.text
        bot.reply_to(message, 'Присылай pdf, docx или txt!')
        return None

    if not user_collections.get(message.from_user.username):
        bot.reply_to(message,
                     'У тебя нет активного файла! Посмотри доступные с помощью /view_files и выбери нужный тебе с помощью /set_file')
        return None

    try:
        ans = ask_question(user_collections[message.from_user.username], message.text)

        bot.send_message(message.from_user.id, ans)

    except:
        bot.reply_to(message, traceback.format_exc())


with open('bot_info.json', 'r') as f:
    d = json.load(f)
    users = d.get('users') or ['tcarroflan']
    admins = d.get('admins') or ['tcarroflan']
    user_collections = d.get('user_collections') or {}
    collections = d.get('collections') or []


bot.polling(none_stop=True, interval=0)