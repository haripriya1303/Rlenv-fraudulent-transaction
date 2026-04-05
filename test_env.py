import asyncio
from client import FraudEnv

async def main():
    print("Connecting...")
    env = FraudEnv(base_url="http://localhost:8000")
    print("Reseting...")
    res = await env.reset()
    print(f"Responded: {res}")
    await env.close()

if __name__ == "__main__":
    asyncio.run(main())
