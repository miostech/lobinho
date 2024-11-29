import telethon.client
from telethon import TelegramClient, sync
import logging
import json
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from openai import OpenAI

# Configuration of OpenAI
clientOpenApi = OpenAI(
    api_key="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",
)

# Configuration of Firebase
cred = credentials.Certificate("lobinho-de-wall-street-firebase-adminsdk-r4x0i-b99f0314cb.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Configuration of Telegram
api_id = 20714327
api_hash = '0d99258beafe83b4afa447d106311819'

client = TelegramClient('session_name', api_id, api_hash)
client.start()


def process_message(message) -> str:
    stream = clientOpenApi.beta.threads.create_and_run(
        assistant_id="asst_iHeTMQZEAG3sfJ9NWhCJS5yn",
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
    if message["original_reply_to_msg_id"]:
        reply_message = client.get_messages(original_channel, ids=message["original_reply_to_msg_id"])
        if reply_message:
            get_message = (db.collection('messages').where("original_id", "==", message["original_reply_to_msg_id"]).where("url_channel", "==", message["url_channel"]).get())
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


def get_messages(url_channel: str, url_channel_to_send: str):
    channel = client.get_entity(url_channel)
    channel_to_send = client.get_entity(url_channel_to_send)

    msgs = client.iter_messages(channel, limit=50, reverse=False)
    # order by date
    msgs = sorted(msgs, key=lambda x: x.date)

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
                .where("url_channel", "==", url_channel).get()):
            continue

        if message.text is None or message.text == "":
            continue
        print("Processing message")
        message_processed = process_message(message.text)
        message_processed = json.loads(message_processed)
        save['message'] = message_processed["msg"]
        save['type'] = message_processed["type"]

        if save['type'] == "prevision" or save['type'] == "update":
            print("Sending message")
            send_message_with_reply(save, save["message"], channel_to_send, channel)


def __main__():
    # the core of the program is, get the messages from the channels and send to another channel
    get_messages('https://t.me/Bybit_Crypto_Signale', 'https://t.me/testemeli')


__main__()
