# langchain、model
from langchain_openai import AzureChatOpenAI    
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# system
import sys
import os
import time
import requests
import json
import base64
from typing import Dict, Any, Optional, List
import random
import asyncio
from dotenv import load_dotenv
from datetime import datetime
# 專案項目
# 獲取專案根目錄的路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Generic image generation prompt - no hardcoded categories
GENERIC_IMAGE_PROMPT = """Create a detailed image prompt for FluxDev based on the provided description. 
Be creative and visually compelling while staying true to the content description. 
Include specific details like camera angles, lighting conditions, visual style, and composition. 
Generate a still image with no motion. Make the prompt detailed and descriptive."""


load_dotenv()
api_base = os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")

def generate_image_prompt_fun(description:str) -> str:
    """
    生成圖片prompt
    Args:
        description (str): 描述
    Returns:
        str: image_prompt
    """
    # system_prompt1 = image_first_prompt
    # system_prompt2 = image_second_prompt
    llm = AzureChatOpenAI(
        azure_endpoint=api_base,
        api_key=api_key,
        azure_deployment="gpt-4o-testing",
        api_version="2025-01-01-preview",
        temperature=0.6,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    generate_first_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", GENERIC_IMAGE_PROMPT),
            ("user", "Specific scene description: {description}")
        ]
    )
    generate_first_prompt_chain = generate_first_prompt_template | llm | StrOutputParser()
    prompt = generate_first_prompt_chain.invoke({"description": description })
    if prompt:
        prompt = prompt.replace("```json", "").replace("```", "")
    # generate_second_prompt_template = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", system_prompt2),
    #     ]
    # )
    # generate_second_prompt_chain = generate_second_prompt_template | llm | StrOutputParser()
    # second_prompt = generate_second_prompt_chain.invoke({"first_answer": first_prompt })
    return prompt

def call_image_request_function(prompt: str, path_name:str) -> Optional[str]:
    """
    使用 ComfyUI 生成圖片
    
    參數:
        prompt (str): prompt_input
        path_name (str): 路徑名稱
    返回:
        Optional[str]: 生成的圖片的 base64 字符串，失敗則返回 None
    """
    try:
        # ComfyUI 的 API 端點

        # COMFY_URL = "http://localhost:8000/" # TODO: 改成要用的url，格式：https://da07-185-219-141-17.ngrok-free.app/api/prompt
        COMFY_URL = "https://7fd6781ec07e.ngrok-free.app/api/prompt"
        if COMFY_URL:
            workflow_path = [
                "workflows/KreaGen.json",
            ]
            # 隨機選擇一個 workflow_path
            selected_path = random.choice(workflow_path)
            with open(f"{selected_path}", "r", encoding='utf-8') as file:
                workflow = json.load(file)

            # Process workflow nodes to replace placeholders
            for node in workflow.values():
                # Replace prompt text in CLIPTextEncode nodes
                if node.get("class_type") == "CLIPTextEncode":
                    if "inputs" in node and "text" in node["inputs"]:
                        # Replace the entire text with the generated prompt
                        node["inputs"]["text"] = prompt
                
                # Update SaveImageExtended to use custom folder name and simpler filename
                if node.get("class_type") == "SaveImageExtended":
                    if "inputs" in node:
                        # Set custom folder name
                        if "foldername_keys" in node["inputs"]:
                            node["inputs"]["foldername_keys"] = path_name
                        # Simplify filename to avoid length issues
                        if "filename_prefix" in node["inputs"]:
                            node["inputs"]["filename_prefix"] = f"{path_name}_%F_%H-%M-%S"
                        # Disable tagger-based filename keys to prevent tags in filename
                        if "filename_keys" in node["inputs"]:
                            node["inputs"]["filename_keys"] = ""  # Remove tagger reference from filename

                # Support core SaveImage node by setting subfolder/filename_prefix
                if node.get("class_type") == "SaveImage":
                    if "inputs" in node:
                        if "subfolder" in node["inputs"]:
                            node["inputs"]["subfolder"] = path_name
                        if "filename_prefix" in node["inputs"]:
                            node["inputs"]["filename_prefix"] = f"{path_name}/{path_name}_%F_%H-%M-%S"
            
            print(f"Processing workflow with prompt length: {len(prompt)} characters")
            print(f"Output folder: {path_name}")
            # 發送請求執行 workflow
            response = requests.post(COMFY_URL, json={"prompt":workflow})
            
            if response.status_code != 200:
                print(f"錯誤: ComfyUI 請求失敗，狀態碼: {response.status_code}, {response.text}")
                return None
            
            # 獲取排隊ID
            prompt_id = response.json().get('prompt_id')
            if not prompt_id:
                print("錯誤: 未能獲取prompt_id")
                return None
            return prompt_id
        
        else:
            return "test_prompt_id"
            
        # # 等待圖片生成完成
        # history_url = f"https://0691-94-140-8-49.ngrok-free.app/history/{prompt_id}"
        # max_attempts = 30  # 最大等待次數
        # attempt = 0
        
        # while attempt < max_attempts:
        #     history_response = requests.get(history_url)
        #     if history_response.status_code == 200:
        #         history_data = history_response.json()
        #         if history_data.get('status', {}).get('completed', False):
        #             print("圖片生成完成")
        #             return "success"  # 或返回其他所需信息
            
        #     await asyncio.sleep(2)  # 等待2秒後再次檢查
        #     attempt += 1
        
        # print("圖片生成超時")
        # return None
    
    except Exception as e:
        print(f"發送生成圖片請求時發生錯誤: {str(e)}")
        return None

def get_tags_and_file_info_from_comfy_history(prompt_id: str, max_attempts: int = 10) -> Dict[str, Optional[str]]:
    """
    Retrieve WD14 Tagger tags and file information from ComfyUI history after image generation
    
    Args:
        prompt_id (str): The prompt ID returned from ComfyUI
        max_attempts (int): Maximum number of attempts to check for completion
        
    Returns:
        Dict[str, Optional[str]]: Dictionary with 'tags', 'file_name', and 'file_path'
    """
    try:
        # Extract base URL from COMFY_URL
        COMFY_URL = "https://7fd6781ec07e.ngrok-free.app/api/prompt"
        base_url = COMFY_URL.replace("/api/prompt", "")
        history_url = f"{base_url}/history/{prompt_id}"
        
        for attempt in range(max_attempts):
            try:
                history_response = requests.get(history_url)
                if history_response.status_code == 200:
                    history_data = history_response.json()
                    
                    # Check if the workflow is complete
                    prompt_data = history_data.get(prompt_id, {})
                    outputs = prompt_data.get('outputs', {})
                    
                    result = {
                        'tags': None,
                        'file_name': None,
                        'file_path': None
                    }
                    
                    # Look for WD14 Tagger output (node "39" in our workflow)
                    if '39' in outputs:
                        tagger_output = outputs['39']
                        if 'tags' in tagger_output:
                            tags = tagger_output['tags'][0] if isinstance(tagger_output['tags'], list) else tagger_output['tags']
                            result['tags'] = tags
                    
                    # Look for SaveImage node outputs to get file information
                    for node_id, output in outputs.items():
                        if 'images' in output:
                            images = output['images']
                            if images and len(images) > 0:
                                image_info = images[0]
                                file_name = image_info.get('filename', '')
                                if file_name:
                                    # Remove extension for file_name
                                    result['file_name'] = os.path.splitext(file_name)[0]
                                    
                                    # Construct file path
                                    subfolder = image_info.get('subfolder', '')
                                    if subfolder:
                                        result['file_path'] = f"C:\\Users\\x7048\\Documents\\ComfyUI\\output\\{subfolder}\\{file_name}"
                                    else:
                                        result['file_path'] = f"C:\\Users\\x7048\\Documents\\ComfyUI\\output\\{file_name}"
                                    break
                    
                    # If we got at least one piece of information, return it
                    if result['tags'] or result['file_name'] or result['file_path']:
                        return result
                
                # Wait before next attempt
                time.sleep(2)
                
            except Exception as e:
                print(f"Error checking history (attempt {attempt + 1}): {str(e)}")
                time.sleep(2)
                
        print(f"Could not retrieve complete info for prompt_id {prompt_id} after {max_attempts} attempts")
        return {
            'tags': None,
            'file_name': None,
            'file_path': None
        }
        
    except Exception as e:
        print(f"Error retrieving info from ComfyUI history: {str(e)}")
        return {
            'tags': None,
            'file_name': None,
            'file_path': None
        }

def batch_generate_images(num_images: int = 10, append_to_file: str = None, theme: str = "fantasy_adventure") -> List[str]:
    """
    批量生成圖片，生成指定數量的不同prompt並調用ComfyUI生成圖片
    
    Args:
        num_images (int): 要生成的圖片數量，默認10張
        append_to_file (str): 要附加到的現有JSON文件名，如果為None則創建新文件
        theme (str): 主題類型，"fantasy_adventure" 或 "magical_quest" 或 "epic_battle"
    
    Returns:
        List[str]: 所有生成的prompt_id列表
    """
    # 不同的奇幻冒險場景描述，根據主題選擇
    fantasy_adventure_scenarios = [
        "Ancient dragon's lair with treasure hoard and magical artifacts",
        "Enchanted forest with glowing mushrooms and mystical creatures",
        "Medieval castle on floating island with waterfalls cascading down",
        "Wizard's tower with swirling magical energy and floating books",
        "Dwarven forge deep in mountain with molten metal and ancient runes",
        "Elven city built in giant trees with bridges connecting platforms",
        "Crystal cave with magical formations and ethereal light sources",
        "Desert oasis with ancient ruins and mysterious portal",
        "Flying ship navigating through storm clouds and lightning",
        "Underground kingdom with glowing crystals and subterranean rivers"
    ]
    
    magical_quest_scenarios = [
        "Hero's journey through mystical portal with glowing runes",
        "Magical sword in stone with ancient prophecies and destiny",
        "Wizard's spell casting with elemental magic and energy",
        "Fantasy battle scene with dragons and knights in epic combat",
        "Enchanted library with floating books and magical knowledge",
        "Mythical creature encounter in misty mountain pass",
        "Magical potion brewing with colorful ingredients and steam",
        "Ancient temple with hidden treasures and guardian spirits",
        "Flying mount soaring over fantasy landscape with castles",
        "Magical transformation scene with light and energy effects"
    ]
    
    epic_battle_scenarios = [
        "Epic battle between armies of light and darkness on vast battlefield",
        "Hero facing ancient evil in dramatic confrontation with magical weapons",
        "Siege of fantasy castle with catapults and magical defenses",
        "Naval battle between pirate ships and royal fleet with sea monsters",
        "Aerial combat between flying creatures and airships",
        "Underground war between dwarves and goblins in caverns",
        "Magical duel between powerful wizards with elemental forces",
        "Cavalry charge across fantasy plains with banners flying",
        "Defense of magical barrier against dark forces",
        "Final showdown in ancient arena with cheering crowds"
    ]
    
    day_trading_scenarios = [
        "Multiple monitor setup showing real-time stock charts and trading platforms",
        "Day trader analyzing candlestick patterns on high-resolution displays",
        "Trading desk with hot keys keyboard and professional trading software",
        "Intraday price movements and volume indicators on trading screens",
        "High-frequency trading setup with millisecond execution displays",
        "Day trading workspace with profit/loss tracking and risk management tools",
        "Real-time market data feeds and news terminals for active trading",
        "Scalping setup with tick charts and order book depth analysis",
        "Professional day trading office with multiple asset class monitors",
        "Technical analysis workspace with drawing tools and trend indicators",
        "Options trading platform showing Greeks and volatility surfaces",
        "Cryptocurrency day trading setup with multiple exchange interfaces",
        "Forex day trading station with currency pair charts and economic calendar",
        "Momentum trading setup tracking breakouts and volume surges",
        "Day trading risk management dashboard with position sizing tools",
        "Pre-market analysis workspace with gap scanners and news feeds",
        "Swing trading transition setup for overnight position management",
        "Pattern day trader workspace meeting PDT rule requirements",
        "Day trading journal software tracking performance metrics",
        "Live trading room setup with chat feeds and mentor guidance"
    ]
    
    bear_market_scenarios = [
        "Red declining stock charts showing steep market losses and downward trends",
        "Trading floor in crisis with panicked expressions and falling market indicators",
        "Bear market symbols with downward trending arrows and negative graphs",
        "Financial stress situation with portfolio losses displayed on screens",
        "Market crash indicators showing massive sell-offs and declining values",
        "Bearish cryptocurrency displays with steep price drops and red candles",
        "Investment portfolio showing significant losses and negative returns",
        "Economic recession indicators on financial news displays and terminals",
        "Risk management crisis meeting with defensive investment strategies",
        "Market volatility displays showing high VIX and fear index readings",
        "Bear market rally signs with temporary upswings in declining trends",
        "Financial media coverage of market corrections and economic uncertainty",
        "Investment advisory meeting discussing defensive positioning strategies",
        "Bear market protection strategies with hedging tools and put options",
        "Economic indicators showing recession signals and declining GDP data",
        "Financial planning during market downturns with conservative approaches",
        "Bear market psychology analysis with fear and greed index displays",
        "Defensive investment sectors performing during economic downturns",
        "Market timing discussions during bear market phases and corrections",
        "Long-term investment perspective during temporary market declines"
    ]
    
    crypto_scenarios = [
        "Bitcoin trading setup with multiple cryptocurrency exchange interfaces",
        "Blockchain technology visualization with distributed ledger networks",
        "Cryptocurrency mining farm with high-performance GPU arrays",
        "DeFi protocol interface showing liquidity pools and yield farming",
        "NFT marketplace display with digital art collections and trading",
        "Crypto wallet security setup with hardware wallets and private keys",
        "Altcoin analysis workspace with technical indicators and market data",
        "Cryptocurrency news terminal showing market sentiment and price alerts",
        "Smart contract development environment with Solidity code editors",
        "Crypto portfolio tracking dashboard with diversified digital assets",
        "Bitcoin ATM machine in modern urban financial district setting",
        "Cryptocurrency conference presentation on blockchain innovation",
        "Digital asset custody solutions with institutional-grade security",
        "Crypto derivatives trading platform with futures and options",
        "Stablecoin ecosystem display showing USDC, USDT, and DAI pairs",
        "Web3 development workspace with decentralized application interfaces",
        "Cryptocurrency regulatory compliance office with legal frameworks",
        "Bitcoin halving event analysis with supply reduction charts",
        "Layer 2 scaling solutions display with Lightning Network interfaces",
        "Crypto tax preparation workspace with transaction tracking software"
    ]
    
    real_estate_scenarios = [
        "Modern luxury home interior with open floor plan and premium finishes",
        "Real estate agent presenting property portfolio on digital tablet",
        "Commercial office building lobby with marble floors and glass elevators",
        "Residential neighborhood aerial view showing property development",
        "Real estate investment analysis workspace with property valuation reports",
        "Property management office with tenant screening and lease agreements",
        "Home staging setup with furniture and decorative elements",
        "Real estate market analysis dashboard with price trends and forecasts",
        "Mortgage application meeting with loan documents and calculators",
        "Property inspection checklist with professional assessment tools",
        "Real estate photography setup capturing interior and exterior shots",
        "Investment property renovation planning with contractor estimates",
        "Real estate auction environment with bidding paddles and displays",
        "Property appraisal workspace with comparable sales analysis",
        "Real estate CRM system managing client relationships and listings",
        "Commercial real estate lease negotiation in modern conference room",
        "Residential property showing with virtual tour technology setup",
        "Real estate investment trust portfolio analysis on financial screens",
        "Property tax assessment documents with municipal evaluation reports",
        "Real estate development site with architectural plans and permits",
        "Home buyer consultation with first-time purchase guidance materials",
        "Property maintenance scheduling system with repair work orders",
        "Real estate market research with demographic and economic data",
        "Rental property income tracking with cash flow projections",
        "Real estate closing ceremony with keys and signed contracts",
        "Property insurance evaluation with coverage assessment documents",
        "Real estate marketing materials with professional photography portfolio",
        "Investment property analysis with cap rates and ROI calculations",
        "Real estate legal documents review with title searches and deeds",
        "Property management software dashboard with maintenance requests",
        "Real estate networking event with industry professionals gathering",
        "Home equity analysis with property value appreciation charts",
        "Real estate crowdfunding platform showing investment opportunities",
        "Property flipping project planning with budget and timeline sheets",
        "Real estate mentorship meeting with experienced investor guidance",
        "Commercial property due diligence with financial and legal reviews",
        "Real estate podcast recording setup discussing market trends",
        "Property wholesaling workspace with contract assignments and deals",
        "Real estate education seminar with investment strategy presentations",
        "Vacation rental property management with booking and guest services",
        "Real estate technology demonstration with AR property visualization",
        "Property asset management meeting with portfolio optimization strategies",
        "Real estate market timing analysis with buying and selling indicators",
        "International real estate investment with global property portfolios",
        "Real estate partnership formation with joint venture agreements",
        "Property syndication presentation with investor group meetings",
        "Real estate tax strategy planning with depreciation and 1031 exchanges",
        "Luxury real estate showcase with high-end property presentations",
        "Real estate data analytics workspace with predictive modeling tools",
        "Property acquisition financing with bank loan application processes"
    ]
    
    global_market_scenarios = [
        "London Stock Exchange trading floor with international market displays",
        "Tokyo financial district skyline during morning market opening hours",
        "Frankfurt banking headquarters with European Central Bank connections",
        "Hong Kong trading terminals showing Asian market indicators",
        "New York Federal Reserve building with global monetary policy displays",
        "Singapore financial hub with multi-currency trading platforms",
        "Swiss private banking office with international wealth management",
        "Dubai International Financial Centre with Middle Eastern market data",
        "Shanghai Stock Exchange with Chinese market capitalizations",
        "Toronto Stock Exchange displaying Canadian commodity markets",
        "Sydney financial district with Australian mining stock indicators",
        "Mumbai stock exchange floor with Indian market growth charts",
        "São Paulo financial center with Brazilian emerging market data",
        "Moscow exchange terminals showing Russian energy sector stocks",
        "Mexico City financial district with Latin American market indicators",
        "Seoul trading floor with Korean technology stock displays",
        "Tel Aviv stock exchange with Israeli innovation sector data",
        "Johannesburg financial center with African mining stock indicators",
        "Global foreign exchange trading desk with major currency pairs",
        "International commodities trading floor with gold and oil prices",
        "Cross-border investment analysis with emerging market portfolios",
        "Global economic summit with international financial ministers",
        "Multinational corporation boardroom with worldwide market analysis",
        "International banking conference with global regulatory discussions",
        "World Bank headquarters with development finance project displays",
        "International Monetary Fund meeting room with global crisis management",
        "G20 finance summit with international economic cooperation",
        "Global investment fund office with diversified country allocations",
        "International trade finance desk with export-import documentation",
        "Offshore banking center with international tax optimization",
        "Global macro hedge fund with worldwide economic trend analysis",
        "International pension fund meeting with global asset allocation",
        "Sovereign wealth fund office with strategic international investments",
        "Global private equity firm with cross-border acquisition analysis",
        "International venture capital meeting with worldwide startup funding",
        "Cross-border mergers and acquisitions advisory workspace",
        "Global supply chain finance with international trade settlements",
        "International insurance company with worldwide risk assessment",
        "Global real estate investment with international property markets",
        "Multinational tax advisory with international compliance frameworks",
        "Global financial technology hub with international fintech solutions",
        "International development bank with emerging market infrastructure",
        "Global carbon credit trading with international climate finance",
        "International arbitrage trading desk with cross-market opportunities",
        "Global wealth management with international client portfolios",
        "International Islamic finance center with Sharia-compliant investments",
        "Global impact investing with international sustainable development",
        "International crisis management center with global market stability",
        "Global financial intelligence unit with international compliance monitoring",
        "International central bank cooperation with global monetary coordination"
    ]
    
    strategies_scenarios = [
        "Chess pieces arranged on marble surface representing strategic financial planning",
        "Golden scales balancing different investment symbols in dramatic lighting",
        "Maze pathway made of paper money leading to financial freedom",
        "Abstract geometric shapes representing portfolio diversification strategies",
        "Compass pointing towards multiple financial goal destinations",
        "Bridge made of coins spanning across a river of opportunity",
        "Tree with branches growing different currency leaves in autumn colors",
        "Hourglass filled with gold coins measuring time value of money",
        "Ladder made of financial documents ascending towards success",
        "Shield protecting valuable assets from market storms",
        "Puzzle pieces forming complete financial strategy picture",
        "Mountain peak representing long-term investment summit achievement",
        "Garden with different investment plants growing at various stages",
        "Lighthouse guiding ships through volatile market waters",
        "Key unlocking treasure chest filled with diversified investments",
        "Balance beam with risk and reward weights in equilibrium",
        "Roadmap with multiple financial milestone markers along the path",
        "Fortress walls protecting wealth from external financial threats",
        "Seed sprouting into money tree representing compound growth",
        "Clock mechanism with gears representing systematic investment timing",
        "Arrow splitting into multiple successful investment directions",
        "Foundation stones building towards financial security structure",
        "Telescope focused on distant financial goal constellation",
        "Blueprint drawings for constructing wealth building strategies",
        "Domino effect showing positive financial momentum chain reaction",
        "Fishing net catching various investment opportunities from market ocean",
        "Pyramid structure with different asset classes forming stable base",
        "Umbrella protecting against financial rain and market downpours",
        "Magnifying glass examining fine details of investment strategies",
        "Stepping stones crossing stream towards financial independence",
        "Lock and key representing secured investment protection strategies",
        "Windmill harnessing market forces for consistent returns",
        "Anchor keeping portfolio stable during turbulent market conditions",
        "Map with treasure X marking optimal investment allocation spots",
        "Prism splitting white light into spectrum of investment options",
        "Seesaw demonstrating risk-return balance in strategic planning",
        "Funnel channeling market opportunities into focused investment strategy",
        "Gear system showing interconnected financial planning mechanisms",
        "Thermometer measuring market temperature for strategic timing",
        "Crystal ball revealing future financial possibilities and outcomes",
        "Phoenix rising from ashes representing financial recovery strategies",
        "Iceberg showing hidden depths of comprehensive financial planning",
        "Spider web catching profitable opportunities in strategic pattern",
        "Pendulum swinging between conservative and aggressive investment approaches",
        "Kaleidoscope creating patterns from diverse investment fragments",
        "Lever system demonstrating financial leverage and strategic positioning",
        "Constellation map connecting financial goals with strategic pathways",
        "Origami crane folded from dollar bills representing transformation",
        "Sundial casting shadows on investment timing decisions",
        "Butterfly emerging from cocoon symbolizing financial metamorphosis"
    ]
    
    risk_vs_reward_scenarios = [
        "Tightrope walker balancing on golden rope over deep canyon with treasure chest",
        "Double-sided coin spinning in mid-air showing opportunity and danger",
        "Mountain climber ascending steep cliff with golden summit in misty clouds",
        "Dice rolling on marble table with diamonds and coal as possible outcomes",
        "Scales weighing feathers against gold bars in dramatic chiaroscuro lighting",
        "Butterfly wings showing fragile beauty next to sharp thorns with roses",
        "Bridge crossing turbulent river with safe shore and treasure island visible",
        "Volcano crater containing both molten danger and precious gemstone deposits",
        "Storm clouds parting to reveal rainbow leading to pot of gold",
        "Spider web glistening with dewdrops, both trap and artistic masterpiece",
        "Ice formations creating both slippery danger and crystalline beauty",
        "Fire consuming old growth while new green shoots emerge from ashes",
        "Ocean waves showing both destructive power and life-giving potential",
        "Glass orb containing swirling storm with golden light at center",
        "Thorny rose bush with magnificent blooms protected by sharp defenses",
        "Cliff edge overlooking vast valley with eagles soaring below",
        "Mirror maze reflecting both escape routes and dead-end traps",
        "Pendulum swinging between two containers of sand and gold",
        "Lightning striking tree creating both destruction and new growth patterns",
        "Hourglass with sand falling through narrow opening, time as risk factor",
        "Yin-yang symbol made from contrasting materials like coal and diamond",
        "Seesaw with heavy anchor on one side, helium balloons on other",
        "Chess board with pieces positioned in high-stakes endgame scenario",
        "Trapeze artists mid-flight with safety net visible far below",
        "Waterfall cascading over precious stones into unknown depths",
        "Hot air balloon floating between storm clouds and sunny skies",
        "Maze built from playing cards with exit paths marked in gold",
        "Compass needle pointing between magnetic north and treasure location",
        "Butterfly collector's net poised over rare specimen in dangerous terrain",
        "Seed pods opening to release both poisonous and beneficial elements",
        "Crystal cave with beautiful formations and unstable ceiling structure",
        "Lighthouse beam cutting through fog toward both safe harbor and rocky shores",
        "Balanced rocks creating natural sculpture that could topple any moment",
        "Golden apple hanging from branch over precipice, requiring dangerous reach",
        "Prism refracting light into both blinding intensity and beautiful spectrum",
        "Wind chimes creating harmony while weathering destructive storm forces",
        "Iceberg showing small visible portion above vast hidden mass below",
        "Labyrinth with multiple paths leading to rewards and dead ends",
        "Telescope focused on distant star that could be salvation or supernova",
        "Garden gate opening to paradise garden with warning signs posted",
        "Tidal pool ecosystem balancing delicate life with crushing wave power",
        "Aerial silk performer suspended between safety platform and void",
        "Phoenix feather glowing with regenerative power amid smoldering ruins",
        "Crystal ball showing multiple possible futures in swirling mists",
        "Rope bridge spanning chasm with beautiful valley visible far below",
        "Magnifying glass focusing sunlight into both harmful ray and beneficial warmth",
        "Kaleidoscope creating beauty from fragments of broken valuable materials",
        "Weather vane pointing between approaching storm and clearing skies",
        "Quicksand pit surrounded by oasis with palm trees and fresh water",
        "Diamond mine shaft descending into darkness with promise of riches"
    ]
    
    astrology_scenarios = [
        "Aries constellation with ram symbolism and fire elements",
        "Taurus constellation with bull imagery and earth elements", 
        "Gemini constellation with twin symbolism and air elements",
        "Cancer constellation with crab imagery and water elements",
        "Leo constellation with lion symbolism and fire elements",
        "Virgo constellation with maiden imagery and earth elements",
        "Libra constellation with scales symbolism and air elements", 
        "Scorpio constellation with scorpion imagery and water elements",
        "Sagittarius constellation with archer symbolism and fire elements",
        "Capricorn constellation with goat imagery and earth elements",
        "Aquarius constellation with water bearer symbolism and air elements",
        "Pisces constellation with fish imagery and water elements",
        "Solar eclipse with mystical astrological significance",
        "Lunar eclipse with celestial astrological meaning",
        "Mercury retrograde with cosmic communication effects",
        "Venus transit with love and beauty astrological influence",
        "Mars opposition with warrior energy and conflict",
        "Jupiter conjunction with expansion and abundance",
        "Saturn return with life lessons and transformation",
        "Uranus square with sudden change and revolution",
        "Neptune trine with dreams and spiritual awakening",
        "Pluto transformation with death and rebirth cycles",
        "Full moon in different zodiac signs with ritual significance",
        "New moon manifestation with astrological timing",
        "Planetary alignment with cosmic significance",
        "Astrological birth chart with natal planet positions",
        "Tarot cards combined with astrological symbolism",
        "Crystal formations aligned with zodiac energies",
        "Sacred geometry patterns reflecting celestial movements",
        "Astrological houses with life theme representations",
        "Elemental balance of fire, earth, air, water signs",
        "Cardinal, fixed, mutable sign energy expressions",
        "Astrological aspects and angular relationships",
        "Celestial bodies in retrograde motion effects",
        "Zodiac wheel with seasonal and temporal divisions",
        "Astrological transits and their life influences",
        "Lunar phases and their emotional significance",
        "Solar return charts and annual cycles",
        "Composite charts and relationship astrology",
        "Progressed charts and life evolution timing",
        "Astrological remedies and healing practices",
        "Vedic astrology with Eastern philosophical elements",
        "Chinese astrology with animal year symbolism",
        "Mayan astrology with calendar and cosmic cycles",
        "Celtic astrology with tree and nature symbolism",
        "Egyptian astrology with deity and pyramid imagery",
        "Babylonian astrology with ancient wisdom traditions",
        "Modern psychological astrology with therapeutic themes",
        "Evolutionary astrology with soul purpose exploration",
        "Electional astrology with optimal timing selection"
    ]
    
    if theme == "magical_quest":
        fantasy_scenarios = magical_quest_scenarios
    elif theme == "epic_battle":
        fantasy_scenarios = epic_battle_scenarios
    elif theme == "astrology":
        fantasy_scenarios = astrology_scenarios
    else:
        fantasy_scenarios = fantasy_adventure_scenarios
    
    prompt_ids = []
    generated_data = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"開始批量生成 {num_images} 張圖片...")
    
    for i in range(num_images):
        try:
            # 使用不同的場景描述來生成多樣化的prompt
            scenario = fantasy_scenarios[i % len(fantasy_scenarios)]
            print(f"\n=== 生成第 {i+1} 張圖片 ===")
            print(f"場景描述: {scenario}")
            
            # 生成圖片prompt
            generated_prompt = generate_image_prompt_fun(scenario)
            print(generated_prompt)
            print(f"生成的prompt長度: {len(generated_prompt)} 字符")
            
            # 調用ComfyUI生成圖片
            path_name = f"batch_{theme}"
            prompt_id = call_image_request_function(generated_prompt, path_name)
            
            if prompt_id:
                prompt_ids.append(prompt_id)
                
                # Wait a bit for ComfyUI to process, then try to get tags and file info
                print("Waiting for image generation and tagging to complete...")
                time.sleep(5)  # Give ComfyUI time to generate the image and run the tagger
                
                # Try to retrieve the generated tags and file information
                file_info = get_tags_and_file_info_from_comfy_history(prompt_id)
                
                # 構建JSON數據結構
                item_data = {
                    "task_id": prompt_id,
                    "prompt": generated_prompt,
                    "file_name": file_info['file_name'] if file_info['file_name'] else path_name,
                    "tags": file_info['tags'] if file_info['tags'] else "No tags retrieved"
                }
                
                # Add file_path if we got it
                if file_info['file_path']:
                    item_data["file_path"] = file_info['file_path']
                
                generated_data.append(item_data)
                
                print(f"成功提交，prompt_id: {prompt_id}")
                if file_info['tags']:
                    print(f"Retrieved tags: {file_info['tags'][:100]}..." if len(file_info['tags']) > 100 else f"Retrieved tags: {file_info['tags']}")
                else:
                    print("Tags could not be retrieved")
                
                if file_info['file_name']:
                    print(f"Retrieved file_name: {file_info['file_name']}")
                if file_info['file_path']:
                    print(f"Retrieved file_path: {file_info['file_path']}")
            else:
                print(f"第 {i+1} 張圖片生成失敗")
            
            # 添加短暫延遲避免API限制，已在tag retrieval中等待了5秒
            # time.sleep(1)
            
        except Exception as e:
            print(f"生成第 {i+1} 張圖片時發生錯誤: {str(e)}")
            continue
    
    # 將數據保存到JSON文件
    theme_key = theme.upper()
    
    if append_to_file:
        output_filename = append_to_file
        try:
            # 先讀取現有JSON文件
            try:
                with open(output_filename, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = {}
            
            # 如果主題已存在，添加到現有列表；否則創建新列表
            if theme_key in existing_data:
                existing_data[theme_key].extend(generated_data)
            else:
                existing_data[theme_key] = generated_data
            
            # 保存更新後的數據
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n✅ 批量生成完成！")
            print(f"成功生成: {len(prompt_ids)}/{num_images} 張圖片")
            print(f"數據已附加到JSON文件: {output_filename}")
            
        except Exception as e:
            print(f"附加數據到JSON文件時發生錯誤: {str(e)}")
    else:
        output_filename = f"batch_image_data_{timestamp}.json"
        try:
            # 創建新的JSON數據結構
            json_data = {
                theme_key: generated_data
            }
            
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n✅ 批量生成完成！")
            print(f"成功生成: {len(prompt_ids)}/{num_images} 張圖片")
            print(f"數據已保存到JSON文件: {output_filename}")
            
        except Exception as e:
            print(f"保存數據到JSON文件時發生錯誤: {str(e)}")
    
    return prompt_ids

if __name__ == "__main__":
    # 批量生成50張fantasy_adventure主題圖片，創建新文件
    batch_generate_images(50, append_to_file=None, theme="astrology")