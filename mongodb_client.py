from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

client = AsyncIOMotorClient(os.getenv("MONGO_URI"))

db = client["MusicCollection"]
collection = db["Songs"]

if __name__ == "__main__":
    async def test_connection():
        try:
            await client.admin.command("ping")
            print("Connected to MongoDB!")
        except Exception as e:
            print(f"Connection failed: {e}")

    async def main():
        try:
            result = await collection.update_one(
                {"_id": "test doc_id2"},
                {"$push": {"metadata": "String2"}},
                upsert=True
                )
            print("Connected to MongoDB!")

        except Exception as e:
            print(f"Error: {e}")
    # asyncio.run(test_connection)
    asyncio.run(main())

