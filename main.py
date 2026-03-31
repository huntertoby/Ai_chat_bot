import discord
from discord.ext import commands
import os
import time  # ✅ 新增時間模組
from dotenv import load_dotenv

from modules import llm_engine
from memory import memory_db

load_dotenv()
os.environ["PATH"] += os.pathsep + os.getcwd()

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)

# ✅ 建立一個暫存大家發言的緩衝池 (被動模式用)
passive_buffers = {}
PASSIVE_TRIGGER_COUNT = 5

# 🌟 新增：注意力追蹤字典 {channel_id: last_active_timestamp}
active_channels = {}
# 🌟 新增：注意力維持時間 (秒)，180秒 = 3分鐘 (可自行修改)
ATTENTION_TIMEOUT = 180


# ================= 🎭 事件處理區 =================

@bot.event
async def on_ready():
    print(f'🤖 Bot 已登入為：{bot.user}')


@bot.event
async def on_message(message):
    print(f"原始訊息: {message.content}")

    # 避免 Bot 自己回覆自己
    if message.author == bot.user:
        return

    # 💡 【優化】把指令判斷移到最前面。如果是 Discord 指令 (如 !記憶)，直接執行並中斷，不要去干擾 LLM
    if message.content.startswith(bot.command_prefix):
        await bot.process_commands(message)
        return

    user_id = str(message.author.id)
    user_name = message.author.display_name
    channel_id = message.channel.id
    current_time = time.time()  # 取得當前時間

    # 清洗字串
    clean_text = message.clean_content
    bot_display_name = message.guild.me.display_name if message.guild else bot.user.name
    clean_text = clean_text.replace(f"@{bot_display_name}", "").replace("yuki", "").strip()

    if not clean_text and not (bot.user in message.mentions or "yuki" in message.content.lower()):
        return

    # 1. 判斷是否為「明確呼叫」(tag 或 提到名字)
    is_explicit_call = bot.user in message.mentions or "yuki" in message.content.lower()

    # 2. 判斷頻道是否處於「持續注意力」活躍狀態
    is_active_channel = False
    if channel_id in active_channels:
        # 檢查是否在超時時間內
        if current_time - active_channels[channel_id] < ATTENTION_TIMEOUT:
            is_active_channel = True
        else:
            # 已經超時，移除活躍狀態
            del active_channels[channel_id]
            print(f"💤 頻道 {channel_id} 注意力已超時，回到被動模式。")

    # 3. 💡 新增「手動退下」機制：如果在活躍狀態，但玩家說了再見，就強制關閉注意力
    exit_keywords = ["掰掰", "再見", "退下", "去睡覺", "閉嘴"]
    if is_active_channel and any(kw in message.content for kw in exit_keywords):
        del active_channels[channel_id]
        await message.reply("好的，我先去旁邊休息囉！有事再叫我～💤")
        return

    # ================= 🗣️ 主動對話模式 =================
    # 只要是「明確呼叫」或「處於活躍狀態」，就進入主動聊天
    if is_explicit_call or is_active_channel:
        if not clean_text:
            await message.channel.send("找我有事嗎？")
            return

        # ✅ 刷新這個頻道的最後活躍時間
        active_channels[channel_id] = current_time
        print(f"🔥 頻道 {channel_id} 處於活躍狀態，啟動主動對話！")

        async with message.channel.typing():
            reply_text = llm_engine.chat(
                user_id=user_id,
                user_name=user_name,
                user_input=clean_text
            )

        await message.reply(reply_text)

    # ================= 🕵️ 被動觀察模式 =================
    else:
        # 🧠 混合式智慧觸發邏輯 (在背景默默聽)
        vector = llm_engine.get_embedding(clean_text)
        is_highly_relevant = False

        if vector:
            related_old_memories = memory_db.search_memory(vector, top_k=1, threshold=0.40)
            if related_old_memories:
                is_highly_relevant = True
                print(
                    f"🎯 [語意觸發] 發現與舊記憶相關的發言 (距離: {related_old_memories[0]['distance']:.3f})，立刻啟動記憶修正！")

        if is_highly_relevant:
            llm_engine.extract_passive_memory(
                user_id=user_id,
                user_name=user_name,
                chat_log=clean_text
            )
        else:
            if user_id not in passive_buffers:
                passive_buffers[user_id] = []

            passive_buffers[user_id].append(clean_text)
            print(
                f"📥 [新話題緩衝] 目前緩衝池 ({len(passive_buffers[user_id])}/{PASSIVE_TRIGGER_COUNT}): {passive_buffers[user_id]}")

            if len(passive_buffers[user_id]) >= PASSIVE_TRIGGER_COUNT:
                chat_history_text = "\n".join(passive_buffers[user_id])
                print(f"⏱️ [批次觸發] 緩衝池已滿，啟動新情報萃取！")

                llm_engine.extract_passive_memory(
                    user_id=user_id,
                    user_name=user_name,
                    chat_log=chat_history_text
                )
                passive_buffers[user_id] = []


# ================= 指令與啟動區保持不變 =================
@bot.command(name="記憶", aliases=["memories", "allinfo"], help="顯示 Yuki 記住關於大家的所有事情")
async def show_memories(ctx):
    # ... (保持原本內容) ...
    memories = memory_db.get_all_memories()

    if not memories:
        await ctx.reply("🧠 目前 Yuki 的腦海裡還像白紙一樣乾淨喔！多跟我聊天吧～")
        return

    reply_lines = ["**🧠 Yuki 的群組八卦記憶庫：**\n"]
    for m in memories:
        reply_lines.append(f"`[{m['timestamp']}]` {m['text']} *(ID: {m['id']})*")

    full_message = "\n".join(reply_lines)

    if len(full_message) > 1950:
        full_message = full_message[:1900] + "\n\n... (記憶太多啦，這邊先截斷顯示最新的一部分！)"

    await ctx.reply(full_message)


@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return
    raise error


if __name__ == "__main__":
    token = os.getenv('DISCORD_TOKEN')
    if token:
        print("🚀 正在連線到 Discord...")
        bot.run(token)
    else:
        print("❌ 找不到 Token！請檢查 .env")