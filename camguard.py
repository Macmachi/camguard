# -*- coding: utf-8 -*-
"""
# CamGuard_v0.4
# Arnaud Ricci 
# Version : 1.0.1
"""
import configparser
import cv2
import numpy as np
import time
import os
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.dispatcher.filters import Text
from aiogram.types import Message
import datetime
from pathlib import Path
from asyncio import Queue

script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)
config_path = os.path.join(script_dir, 'config.ini')

config = configparser.ConfigParser()
config.read(config_path)

#Défini dans le fichier config.ini
motion_threshold = int(config['DETECTION']['motion_threshold'])
kernel_size = int(config['DETECTION']['kernel_size'])
TELEGRAM_BOT_TOKEN = config['BOT']['telegram_bot_token']
YOUR_CHAT_ID = config['BOT']['your_chat_ID']
AUTH_TOKEN = config['BOT']['auth_token']

motion_detection_enabled = False
motion_detection_command = None

def log_message(message: str):
    with open("camguard-log.log", "a") as log_file:
        log_file.write(f"{datetime.datetime.now()} - {message}\n")
        print(f"{datetime.datetime.now()} - {message}")

async def handle_message(message: types.Message, command_queue):
    log_message("Message reçu")
    global motion_detection_enabled, motion_detection_command
    log_message(f"Statut d'activation : {motion_detection_enabled}")
    chat_id = int(message.chat.id)
    log_message(f"YOUR_CHAT_ID ({YOUR_CHAT_ID}) == Chat ID : {chat_id}")

    if chat_id == int(YOUR_CHAT_ID):
        text = message.text.lower()

        # Vérifiez si le message contient le jeton d'authentification
        if AUTH_TOKEN in text:
            await command_queue.put(text.replace(AUTH_TOKEN, '').strip())
            log_message(f"Commande reçue : {motion_detection_command}")
        else:
            log_message("Jeton d'authentification incorrect")
    else:
        log_message(f"Unauthorized chat ID: {chat_id}")

def detect_mouvement(frame1, frame2):
    if frame1 is None or frame2 is None:
        log_message("Erreur : Impossible de lire l'image de la webcam")
        return
    else:
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)  
        _, threshold = cv2.threshold(blur, motion_threshold, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(threshold, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return len(contours) > 0

def save_video(video_writer, frames):
    for frame in frames:
        video_writer.write(frame)

async def send_detection_alert(bot, file_path, YOUR_CHAT_ID, is_video=False):
    try:
        text = "Quelque chose a été détecté par la caméra"

        with open(file_path, "rb") as file:
            log_message(f"Envoie du message à Telegram")
            await bot.send_message(chat_id=YOUR_CHAT_ID, text=text)
            log_message(f"Envoie de la {('photo', 'vidéo')[is_video]} à telegram")
            if is_video:
                await bot.send_video(chat_id=YOUR_CHAT_ID, video=file)
            else:
                await bot.send_photo(chat_id=YOUR_CHAT_ID, photo=file)
    except Exception as e:
        await on_telegram_api_error(e)

async def on_telegram_api_error(e):
    error_message = "Désolé, une erreur s'est produite lors de l'envoi du message à Telegram. Veuillez réessayer plus tard."
    log_message(f"Erreur lors du traitement du message : {e}")
    log_message(f"Message d'erreur : {error_message}")


async def main_async():
    log_message("Programme démarré")
    print("Programme démarré")

    record_dir = 'videos'
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    bot = Bot(TELEGRAM_BOT_TOKEN)
    dp = Dispatcher(bot)
    command_queue = Queue()
    dp.register_message_handler(lambda message: handle_message(message, command_queue), content_types=[types.ContentType.TEXT])
    log_message("Gestionnaire de messages enregistré")
    # Mettre 0, pour windows mais sur linux on utilise /dev/video0 ou /dev/video1, etc.
    cap = cv2.VideoCapture('/dev/video0')

    log_message("Bot en attente de messages...")

    async def detect_motion_loop(bot, cap, record_dir, YOUR_CHAT_ID, command_queue):
        global motion_detection_enabled, motion_detection_command
        motion_detection_state_changed = False

        _, frame1 = cap.read()
        _, frame2 = cap.read()

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        video_duration = 5  # seconds
        video_frames = fps * video_duration
        frame_buffer = []

        try:
            while True:
                # Récupérez la commande de la queue (ne bloque pas)
                try:
                    command = await asyncio.wait_for(command_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    command = None
                #Dans le chat telegram mettre : Le_Token /enable
                if command == "/enable" and not motion_detection_enabled:
                    motion_detection_enabled = True
                    motion_detection_state_changed = True
                elif command == "/disable" and motion_detection_enabled:
                    motion_detection_enabled = False
                    print("")
                    motion_detection_state_changed = True

                if motion_detection_state_changed:
                    message_text = "Détection de mouvement activée." if motion_detection_enabled else "Détection de mouvement désactivée."
                    log_message(f"Statut de détection de mouvement : {motion_detection_enabled}")
                    log_message(f"Envoie du message telegram : {message_text}")
                    await bot.send_message(chat_id=YOUR_CHAT_ID, text=message_text)
                    motion_detection_state_changed = False

                if motion_detection_enabled and detect_mouvement(frame1, frame2):
                    log_message("Quelque chose a été détecté par la caméra")

                    for _ in range(video_frames):
                        ret, frame_to_save = cap.read()
                        frame_buffer.append(frame_to_save)
                        await asyncio.sleep(1 / fps)

                    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                    file_name = f"{record_dir}/detected_{timestamp}.avi"

                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    height, width, _ = frame_buffer[0].shape
                    video_writer = cv2.VideoWriter(file_name, fourcc, fps, (width, height))

                    save_video(video_writer, frame_buffer)
                    video_writer.release()
                    frame_buffer = []

                    await send_detection_alert(bot, file_name, YOUR_CHAT_ID, is_video=True)
                    await asyncio.sleep(60)

                #Ces deux lignes assurent que le script continue de comparer les images (frames) en temps réel pour détecter les mouvements.
                frame1 = frame2
                ret, frame2 = cap.read()

                if cv2.waitKey(40) == 27:
                    break

        except KeyboardInterrupt:
            pass

        finally:
            log_message("Programme terminé")
            cv2.destroyAllWindows()
            cap.release()

    await asyncio.gather(dp.start_polling(), detect_motion_loop(bot, cap, record_dir, YOUR_CHAT_ID, command_queue))

if __name__ == '__main__':
    asyncio.run(main_async())