import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.docstore.document import Document
import json

# Plant Knowledge Base
PLANT_KNOWLEDGE_BASE = """
TOMATO PLANTS - COMPLETE CARE GUIDE

HEALTHY TOMATO LEAVES:
- Color: Deep green, uniform color
- Texture: Smooth, slightly fuzzy
- Structure: Full, unwilted leaves
- Growth: New leaves emerging regularly
Care Tips: 🌞 Full sun (6-8 hours), 💧 Water deeply but infrequently, 🌡️ Temperature 70-85°F

BACTERIAL SPOT:
- Symptoms: Small dark brown spots with yellow halos on leaves and fruit
- Causes: Xanthomonas bacteria, spread by water splash
- Prevention: Avoid overhead watering, space plants for airflow
Treatment: 🔬 Copper-based fungicides, 🌱 Remove infected leaves, 💧 Drip irrigation recommended
Action: Moderate severity - act within 1 week

EARLY BLIGHT:
- Symptoms: Concentric rings (target-like) brown spots on lower leaves
- Causes: Alternaria fungus, thrives in warm humid conditions
- Prevention: Mulch to prevent soil splash, rotate crops yearly
Treatment: 🍃 Remove affected leaves, 🔬 Organic fungicides (neem oil), 🌾 Improve air circulation
Action: Can spread quickly - treat immediately

LATE BLIGHT:
- Symptoms: Large brown-black water-soaked patches, white mold on undersides
- Causes: Phytophthora infestans, same organism as potato famine
- Prevention: Plant resistant varieties, avoid wet foliage
Treatment: ⚠️ URGENT - Remove entire plant if severe, 🔬 Copper fungicides early, 🔥 Destroy infected material
Action: CRITICAL - can destroy crop in days

LEAF MOLD:
- Symptoms: Yellow patches on upper leaf surface, olive-green mold underneath
- Causes: Passalora fulva fungus, loves high humidity
- Prevention: Greenhouse ventilation, reduce humidity below 85%
Treatment: 🌬️ Increase airflow, 🔬 Fungicide spray, ✂️ Prune dense foliage
Action: Manageable - treat within 2 weeks

SEPTORIA LEAF SPOT:
- Symptoms: Small circular spots with dark borders and gray centers
- Causes: Septoria lycopersici fungus, spreads in wet conditions
- Prevention: Stake plants, mulch soil, water at base
Treatment: 🍂 Remove lower leaves, 🔬 Chlorothalonil fungicides, 🌱 Apply early in season
Action: Progressive disease - start treatment early

SPIDER MITES:
- Symptoms: Fine webbing, yellow stippling on leaves, leaf drop
- Causes: Tetranychus urticae, thrives in hot dry conditions
- Prevention: Maintain humidity, spray foliage regularly
Treatment: 💦 Strong water spray, 🐞 Release predatory mites, 🧴 Insecticidal soap
Action: Can multiply rapidly - treat within days

TARGET SPOT:
- Symptoms: Brown spots with concentric rings, similar to early blight but on all plant parts
- Causes: Corynespora cassiicola fungus
- Prevention: Avoid leaf wetness, space plants properly
Treatment: 🔬 Fungicide rotation, 🌿 Remove infected tissue, 💨 Improve ventilation
Action: Moderate severity - treat within 1 week

YELLOW LEAF CURL VIRUS:
- Symptoms: Upward leaf curling, yellowing, stunted growth
- Causes: Begomovirus spread by whiteflies
- Prevention: Control whiteflies, use resistant varieties, reflective mulch
Treatment: ⚠️ No cure - remove infected plants, 🐛 Control whitefly vectors, 🌱 Plant virus-free seedlings
Action: CRITICAL - remove immediately to prevent spread

MOSAIC VIRUS:
- Symptoms: Mottled yellow-green patterns, distorted leaves, reduced yield
- Causes: Tobacco mosaic virus (TMV), spread by handling
- Prevention: Wash hands, disinfect tools, avoid tobacco products near plants
Treatment: ⚠️ No cure available, 🔥 Remove infected plants, 🧤 Use gloves when handling
Action: CRITICAL - isolate or remove infected plants

GENERAL PLANT CARE PRINCIPLES:
Watering: 💧 Deep watering 1-2 times per week, early morning best
Sunlight: 🌞 6-8 hours direct sun daily for fruiting plants
Soil: 🌱 Well-draining, pH 6.0-6.8, rich in organic matter
Fertilization: 🌿 Balanced NPK every 2-3 weeks during growing season
Pruning: ✂️ Remove suckers, lower leaves for airflow
Pest Prevention: 🐛 Regular inspection, companion planting, beneficial insects

SEASONAL CARE:
Spring: 🌱 Plant after last frost, harden off seedlings, mulch beds
Summer: ☀️ Consistent watering, shade cloth in extreme heat, harvest regularly
Fall: 🍂 Reduce watering, harvest remaining fruit, remove plant debris
Winter: ❄️ Plan next season, clean tools, review crop rotation

ECO-FRIENDLY PRACTICES:
Composting: ♻️ Turn kitchen scraps into nutrient-rich soil
Rainwater Collection: 💧 Use rain barrels for sustainable watering
Natural Pest Control: 🐞 Ladybugs, lacewings, companion planting
Organic Mulch: 🌾 Grass clippings, straw, leaves for moisture retention
"""

class PlantRAGSystem:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.initialized = False
        
    def initialize(self):
        """Initialize the RAG system with embeddings and vector store"""
        try:
            # Create embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Split knowledge base into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            
            chunks = text_splitter.split_text(PLANT_KNOWLEDGE_BASE)
            documents = [Document(page_content=chunk) for chunk in chunks]
            
            # Create vector store
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            
            self.initialized = True
            return True
        except Exception as e:
            print(f"Error initializing RAG system: {e}")
            return False
    
    def get_personalized_advice(self, disease_name, context=""):
        """Get personalized advice for a specific disease"""
        if not self.initialized:
            self.initialize()
        
        query = f"{disease_name} treatment prevention symptoms {context}"
        
        try:
            # Retrieve relevant documents
            docs = self.vectorstore.similarity_search(query, k=3)
            
            # Combine retrieved content
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # Extract key information
            advice = self._extract_advice(context_text, disease_name)
            return advice
        except Exception as e:
            return self._get_fallback_advice(disease_name)
    
    def _extract_advice(self, context, disease_name):
        """Extract structured advice from context"""
        advice = {
            'symptoms': [],
            'causes': [],
            'prevention': [],
            'treatment': [],
            'urgency': 'Moderate',
            'tips': []
        }
        
        lines = context.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if 'Symptoms:' in line:
                current_section = 'symptoms'
                advice['symptoms'].append(line.split('Symptoms:')[1].strip())
            elif 'Causes:' in line:
                current_section = 'causes'
                advice['causes'].append(line.split('Causes:')[1].strip())
            elif 'Prevention:' in line:
                current_section = 'prevention'
                advice['prevention'].append(line.split('Prevention:')[1].strip())
            elif 'Treatment:' in line:
                current_section = 'treatment'
                content = line.split('Treatment:')[1].strip()
                # Extract emoji tips
                if '🌞' in content or '💧' in content or '🔬' in content:
                    advice['tips'].append(content)
                advice['treatment'].append(content)
            elif 'Action:' in line:
                advice['urgency'] = line.split('Action:')[1].strip()
            elif 'Care Tips:' in line:
                advice['tips'].append(line.split('Care Tips:')[1].strip())
            elif current_section and line.startswith('-'):
                advice[current_section].append(line[1:].strip())
        
        return advice
    
    def _get_fallback_advice(self, disease_name):
        """Provide fallback advice if RAG fails"""
        return {
            'symptoms': [f'Visible signs on {disease_name}'],
            'causes': ['Environmental or pathogenic factors'],
            'prevention': ['Maintain good plant hygiene', 'Ensure proper watering', 'Adequate sunlight'],
            'treatment': ['Remove affected parts', 'Apply appropriate fungicide', 'Improve growing conditions'],
            'urgency': 'Moderate - Monitor closely',
            'tips': ['🌞 Ensure adequate sunlight', '💧 Water appropriately', '🌱 Check soil drainage']
        }
    
    def get_quick_tips(self, disease_name):
        """Get quick actionable tips with emojis"""
        advice = self.get_personalized_advice(disease_name)
        
        tips = []
        if advice['tips']:
            tips.extend(advice['tips'])
        else:
            # Generate tips based on urgency
            if 'CRITICAL' in advice['urgency'].upper():
                tips = ['⚠️ Act immediately - disease spreads rapidly', 
                       '🔥 Remove infected plants to prevent spread',
                       '🧤 Use protective equipment when handling']
            elif 'URGENT' in advice['urgency'].upper():
                tips = ['⏰ Treat within 24-48 hours',
                       '🔬 Apply appropriate fungicide/pesticide',
                       '✂️ Remove and destroy affected parts']
            else:
                tips = ['🌞 Ensure 6-8 hours of sunlight',
                       '💧 Water deeply but less frequently',
                       '🌱 Maintain good soil drainage']
        
        return tips[:3]  # Return top 3 tips

# Global instance
rag_system = PlantRAGSystem()