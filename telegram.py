import datetime
import time
import telethon.client
from telethon import TelegramClient, sync
from telethon.errors import FloodWaitError
import logging
import json
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from openai import OpenAI
import pytz

# Configuração do OpenAI
clientOpenApi = OpenAI(
    api_key="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",
)

# Configuração do Firebase
cred = credentials.Certificate("lobinho-de-wall-street-firebase-adminsdk-r4x0i-0a073da784.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Configuração do Telegram
api_id = 20714327
api_hash = '0d99258beafe83b4afa447d106311819'
client = TelegramClient('session_name', api_id, api_hash)
client.start()

# Cache de entidades para evitar chamadas repetitivas
channel_cache = {}


def get_entity_cached(url_channel):
    """Obtém uma entidade de canal com cache para evitar chamadas repetidas."""
    if url_channel not in channel_cache:
        channel_cache[url_channel] = client.get_entity(url_channel)
    return channel_cache[url_channel]


def process_message(message, assistant_id) -> str:
    """Processa a mensagem usando a API do OpenAI."""
    stream = clientOpenApi.beta.threads.create_and_run(
        assistant_id=assistant_id,
        thread={
            "messages": [
                {"role": "user", "content": message}
            ]
        },
        stream=True,
    )
    assistant_response = ""
    for event in stream:
        logging.info(f"Received event: {event}")
        if event.event == "thread.message.delta":
            for content in event.data.delta.content:
                if content.type == "text":
                    assistant_response += content.text.value
    return assistant_response


def send_message_with_reply(message, msg, chn, original_channel):
    """Envia mensagens com respostas preservadas."""
    if message["original_reply_to_msg_id"]:
        reply_message = client.get_messages(original_channel, ids=message["original_reply_to_msg_id"])
        if reply_message:
            get_message = (db.collection('messages')
                           .where("original_id", "==", message["original_reply_to_msg_id"])
                           .where("url_channel", "==", message["url_channel"])
                           .where("url_channel_to_send", "==", message["url_channel_to_send"]).get())
            if get_message:
                client.send_message(chn, msg, reply_to=get_message[0].to_dict()["id"])
                message["reply_to_msg_id"] = get_message[0].to_dict()["id"]
                db.collection('messages').add(message)
                return
            new_id_reply = client.send_message(chn, reply_message.text)
            message_copy = message.copy()
            message_copy["id"] = new_id_reply.id
            message_copy["message"] = reply_message.text
            message_copy["original_id"] = reply_message.id
            db.collection('messages').add(message_copy)
            new_id = client.send_message(chn, msg, reply_to=new_id_reply)
            message["id"] = new_id.id
            message["reply_to_msg_id"] = new_id_reply.id
            db.collection('messages').add(message)
            return
    else:
        new_id = client.send_message(chn, msg)
        message["id"] = new_id.id
        db.collection('messages').add(message)
        return


def get_messages(url_channel: str, url_channel_to_send: str, assistant_id: str, date_init: datetime, limit=10):
    """Obtém mensagens de um canal e envia para outro."""
    try:
        channel = get_entity_cached(url_channel)
        channel_to_send = get_entity_cached(url_channel_to_send)

        msgs = client.iter_messages(channel, limit=limit, reverse=False)
        msgs = sorted(msgs, key=lambda x: x.date)

        dd = date_init
        msgs = [msg for msg in msgs if msg.date > dd]

        for message in msgs:
            save = {
                "id": "",  # id of new message
                "original_id": message.id,  # id of original message
                "reply_to_msg_id": "",  # id of reply message
                "original_reply_to_msg_id": message.reply_to_msg_id,  # id of original reply message
                "message": message.text,  # text of message
                "original_message": message.text,  # text of original message
                "type": "",  # type of message
                "date": message.date,  # date of message
                "url_channel": url_channel,  # url of channel
                "url_channel_to_send": url_channel_to_send  # url of channel to send
            }

            if (db.collection('messages')
                    .where("original_id", "==", message.id)
                    .where("url_channel", "==", url_channel)
                    .where("url_channel_to_send", "==", url_channel_to_send).get()):
                continue

            if not message.text:
                continue
            print("Processing message")
            message_processed = process_message(message.text, assistant_id)
            message_processed = json.loads(message_processed)
            save['message'] = message_processed["msg"]
            save['type'] = message_processed["type"]

            if save['type'] in ["prevision", "update"]:
                print("Sending message")
                send_message_with_reply(save, save["message"], channel_to_send, channel)
            else:
                print("Save message whit out send")
                db.collection('messages').add(save)

    except FloodWaitError as e:
        print(f"FloodWaitError: Aguarde {e.seconds} segundos.")


def __main__():
    """Loop principal do programa."""
    while True:
        try:
            print("Getting messages")
            get_messages(
                'https://t.me/Bybit_Crypto_Signale',
                'https://t.me/+IfvNeSmY_QcxYzBk',
                'asst_QZdbRgRyym253YcRtLQXsGoH',
                datetime.datetime(2024, 11, 25, 23, 00, 0, 00000, tzinfo=pytz.UTC),
                limit=10
            )
            get_messages(
                'https://t.me/Bybit_Crypto_Signale',
                'https://t.me/+bQJ9IKnaIdA1MDRk',
                'asst_iHeTMQZEAG3sfJ9NWhCJS5yn',
                datetime.datetime(2024, 11, 25, 23, 00, 0, 00000, tzinfo=pytz.UTC),
                limit=10
            )
            print("End process")
            time.sleep(60)  # Aguarda 5 minutos entre os ciclos
        except Exception as e:
            logging.error(f"Error processing webhook: {str(e)}")
            time.sleep(60)


__main__()
