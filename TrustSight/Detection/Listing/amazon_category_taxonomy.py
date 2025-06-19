from typing import Tuple, List

class AmazonCategoryTaxonomy:
    """Complete Amazon category taxonomy for accurate product categorization and mismatch detection."""
    
    def __init__(self):
        self.categories = {
            'electronics': {
                'name': 'Electronics',
                'keywords': ['electronic', 'device', 'gadget', 'tech', 'digital'],
                'subcategories': {
                    'computers': ['laptop', 'desktop', 'pc', 'computer', 'notebook', 'chromebook', 'macbook'],
                    'phones': ['phone', 'smartphone', 'iphone', 'android', 'mobile', 'cellular'],
                    'tablets': ['tablet', 'ipad', 'kindle', 'surface'],
                    'cameras': ['camera', 'dslr', 'mirrorless', 'webcam', 'gopro', 'camcorder'],
                    'audio': ['headphone', 'earphone', 'speaker', 'microphone', 'earbuds', 'airpods'],
                    'tv_video': ['tv', 'television', 'monitor', 'projector', 'display', 'screen'],
                    'gaming': ['playstation', 'xbox', 'nintendo', 'console', 'controller', 'gaming'],
                    'accessories': ['cable', 'charger', 'adapter', 'battery', 'case', 'cover', 'mount']
                },
                'incompatible': ['food', 'pet_food', 'plants', 'seeds', 'live_animals']
            },
            
            'clothing': {
                'name': 'Clothing, Shoes & Jewelry',
                'keywords': ['clothing', 'apparel', 'fashion', 'wear', 'outfit'],
                'subcategories': {
                    'mens_clothing': ['mens', 'men', 'shirt', 'pants', 'jacket', 'suit', 'tie'],
                    'womens_clothing': ['womens', 'women', 'dress', 'blouse', 'skirt', 'top'],
                    'kids_clothing': ['kids', 'children', 'boys', 'girls', 'baby'],
                    'shoes': ['shoe', 'sneaker', 'boot', 'sandal', 'heel', 'loafer', 'slipper'],
                    'jewelry': ['ring', 'necklace', 'bracelet', 'earring', 'pendant', 'chain'],
                    'watches': ['watch', 'smartwatch', 'timepiece'],
                    'accessories': ['belt', 'wallet', 'purse', 'handbag', 'scarf', 'hat', 'gloves']
                },
                'incompatible': ['electronics', 'appliances', 'automotive', 'tools', 'industrial']
            },
            
            'home_kitchen': {
                'name': 'Home & Kitchen',
                'keywords': ['home', 'kitchen', 'household', 'domestic'],
                'subcategories': {
                    'furniture': ['chair', 'table', 'sofa', 'bed', 'desk', 'cabinet', 'shelf'],
                    'kitchen': ['cookware', 'pot', 'pan', 'knife', 'utensil', 'appliance'],
                    'bedding': ['mattress', 'pillow', 'sheet', 'blanket', 'comforter', 'duvet'],
                    'bath': ['towel', 'shower', 'bathroom', 'toilet', 'sink'],
                    'decor': ['decoration', 'lamp', 'vase', 'frame', 'artwork', 'rug'],
                    'storage': ['container', 'box', 'organizer', 'basket', 'bin'],
                    'cleaning': ['vacuum', 'mop', 'cleaner', 'detergent', 'sponge']
                },
                'incompatible': ['clothing', 'automotive', 'industrial']
            },
            
            'books': {
                'name': 'Books',
                'keywords': ['book', 'novel', 'read', 'literature', 'publication'],
                'subcategories': {
                    'fiction': ['novel', 'story', 'fiction', 'fantasy', 'mystery', 'romance'],
                    'nonfiction': ['biography', 'history', 'science', 'self-help', 'cookbook'],
                    'textbooks': ['textbook', 'educational', 'academic', 'study'],
                    'childrens': ['children', 'kids', 'picture book', 'young adult'],
                    'ebooks': ['kindle', 'ebook', 'digital book'],
                    'audiobooks': ['audiobook', 'audible', 'audio book']
                },
                'incompatible': ['electronics', 'tools', 'automotive', 'fresh_food']
            },
            
            'sports_outdoors': {
                'name': 'Sports & Outdoors',
                'keywords': ['sports', 'fitness', 'outdoor', 'exercise', 'athletic'],
                'subcategories': {
                    'exercise': ['gym', 'weight', 'yoga', 'fitness', 'workout', 'exercise'],
                    'outdoor': ['camping', 'hiking', 'tent', 'backpack', 'outdoor'],
                    'sports_equipment': ['ball', 'racket', 'bat', 'golf', 'tennis', 'basketball'],
                    'cycling': ['bike', 'bicycle', 'cycling', 'helmet'],
                    'water_sports': ['swimming', 'surf', 'kayak', 'fishing'],
                    'winter_sports': ['ski', 'snowboard', 'ice', 'hockey']
                },
                'incompatible': ['books', 'office_supplies', 'baby_formula']
            },
            
            'health_beauty': {
                'name': 'Health & Personal Care',
                'keywords': ['health', 'beauty', 'personal care', 'wellness'],
                'subcategories': {
                    'vitamins': ['vitamin', 'supplement', 'mineral', 'protein'],
                    'medical': ['medical', 'first aid', 'bandage', 'thermometer'],
                    'beauty': ['makeup', 'cosmetic', 'skincare', 'beauty'],
                    'personal_care': ['shampoo', 'soap', 'toothpaste', 'deodorant'],
                    'health_devices': ['blood pressure', 'glucose', 'oximeter', 'scale']
                },
                'incompatible': ['tools', 'automotive', 'industrial', 'raw_materials']
            },
            
            'grocery': {
                'name': 'Grocery & Gourmet Food',
                'keywords': ['food', 'grocery', 'eat', 'drink', 'consumable'],
                'subcategories': {
                    'fresh': ['fresh', 'fruit', 'vegetable', 'meat', 'dairy'],
                    'pantry': ['canned', 'pasta', 'rice', 'cereal', 'snack'],
                    'beverages': ['coffee', 'tea', 'juice', 'soda', 'water'],
                    'gourmet': ['gourmet', 'specialty', 'organic', 'artisan'],
                    'baby_food': ['formula', 'baby food', 'toddler']
                },
                'incompatible': ['electronics', 'tools', 'automotive', 'clothing']
            },
            
            'automotive': {
                'name': 'Automotive',
                'keywords': ['car', 'auto', 'vehicle', 'automotive', 'truck'],
                'subcategories': {
                    'parts': ['part', 'engine', 'brake', 'filter', 'spark plug'],
                    'accessories': ['seat cover', 'floor mat', 'phone mount', 'dash cam'],
                    'tools': ['wrench', 'jack', 'diagnostic', 'tool'],
                    'care': ['oil', 'wax', 'cleaner', 'polish'],
                    'tires': ['tire', 'wheel', 'rim']
                },
                'incompatible': ['clothing', 'food', 'books', 'baby', 'beauty']
            },
            
            'tools': {
                'name': 'Tools & Home Improvement',
                'keywords': ['tool', 'hardware', 'improvement', 'repair', 'build'],
                'subcategories': {
                    'power_tools': ['drill', 'saw', 'sander', 'grinder', 'power tool'],
                    'hand_tools': ['hammer', 'screwdriver', 'wrench', 'pliers'],
                    'hardware': ['screw', 'nail', 'bolt', 'fastener', 'hinge'],
                    'electrical': ['wire', 'outlet', 'switch', 'electrical'],
                    'plumbing': ['pipe', 'faucet', 'valve', 'plumbing'],
                    'paint': ['paint', 'brush', 'roller', 'primer']
                },
                'incompatible': ['clothing', 'food', 'baby', 'books', 'beauty']
            },
            
            'toys_games': {
                'name': 'Toys & Games',
                'keywords': ['toy', 'game', 'play', 'fun', 'entertainment'],
                'subcategories': {
                    'toys': ['toy', 'doll', 'action figure', 'lego', 'plush'],
                    'games': ['board game', 'card game', 'puzzle', 'game'],
                    'outdoor_play': ['swing', 'slide', 'trampoline', 'playhouse'],
                    'educational': ['learning', 'stem', 'educational toy'],
                    'arts_crafts': ['craft', 'art', 'coloring', 'clay']
                },
                'incompatible': ['tools', 'automotive', 'industrial', 'medical']
            },
            
            'pet_supplies': {
                'name': 'Pet Supplies',
                'keywords': ['pet', 'dog', 'cat', 'animal', 'pet supply'],
                'subcategories': {
                    'dog': ['dog', 'puppy', 'canine'],
                    'cat': ['cat', 'kitten', 'feline'],
                    'fish': ['fish', 'aquarium', 'tank'],
                    'bird': ['bird', 'cage', 'perch'],
                    'small_animal': ['hamster', 'rabbit', 'guinea pig']
                },
                'incompatible': ['human_food', 'clothing', 'electronics', 'beauty']
            },
            
            'baby': {
                'name': 'Baby',
                'keywords': ['baby', 'infant', 'newborn', 'toddler'],
                'subcategories': {
                    'feeding': ['bottle', 'formula', 'baby food', 'high chair'],
                    'diapering': ['diaper', 'wipe', 'changing'],
                    'nursery': ['crib', 'bassinet', 'mobile', 'monitor'],
                    'clothing': ['onesie', 'baby clothes', 'bib'],
                    'travel': ['stroller', 'car seat', 'carrier'],
                    'toys': ['rattle', 'teether', 'baby toy']
                },
                'incompatible': ['tools', 'automotive', 'industrial', 'adult_items']
            },
            
            'office': {
                'name': 'Office Products',
                'keywords': ['office', 'desk', 'work', 'business', 'stationery'],
                'subcategories': {
                    'supplies': ['pen', 'paper', 'stapler', 'tape', 'folder'],
                    'furniture': ['desk', 'chair', 'filing cabinet', 'bookshelf'],
                    'technology': ['printer', 'scanner', 'shredder', 'calculator'],
                    'organization': ['organizer', 'planner', 'calendar', 'label']
                },
                'incompatible': ['food', 'pet_supplies', 'automotive', 'baby']
            },
            
            'garden_outdoor': {
                'name': 'Garden & Outdoor',
                'keywords': ['garden', 'lawn', 'outdoor', 'yard', 'patio'],
                'subcategories': {
                    'gardening': ['plant', 'seed', 'soil', 'fertilizer', 'pot'],
                    'lawn_care': ['mower', 'trimmer', 'sprinkler', 'hose'],
                    'outdoor_living': ['grill', 'patio furniture', 'umbrella', 'fire pit'],
                    'pest_control': ['pesticide', 'trap', 'repellent']
                },
                'incompatible': ['electronics', 'clothing', 'books', 'office']
            },
            
            'industrial': {
                'name': 'Industrial & Scientific',
                'keywords': ['industrial', 'scientific', 'commercial', 'professional'],
                'subcategories': {
                    'lab': ['beaker', 'microscope', 'lab equipment', 'chemical'],
                    'safety': ['safety', 'protective', 'gloves', 'goggles', 'mask'],
                    'janitorial': ['cleaning', 'janitorial', 'commercial cleaning'],
                    'material_handling': ['pallet', 'forklift', 'warehouse'],
                    'fasteners': ['industrial fastener', 'bulk hardware']
                },
                'incompatible': ['toys', 'baby', 'food', 'clothing', 'home_decor']
            }
        }
        
        self._build_lookup_tables()
    
    def _build_lookup_tables(self):
        self.keyword_to_category = {}
        self.subcategory_to_main = {}
        
        for main_cat, data in self.categories.items():
            for keyword in data['keywords']:
                self.keyword_to_category[keyword] = main_cat
            
            for subcat, keywords in data['subcategories'].items():
                self.subcategory_to_main[subcat] = main_cat
                for keyword in keywords:
                    self.keyword_to_category[keyword] = main_cat
    
    def identify_category(self, text: str) -> Tuple[str, List[str]]:
        text_lower = text.lower()
        mentioned_categories = set()
        
        for keyword, category in self.keyword_to_category.items():
            if keyword in text_lower:
                mentioned_categories.add(category)
        
        if mentioned_categories:
            primary = list(mentioned_categories)[0]
            return primary, list(mentioned_categories)
        
        return 'unknown', []
    
    def are_categories_compatible(self, cat1: str, cat2: str) -> bool:
        if cat1 == cat2:
            return True
        
        if cat1 in self.categories:
            if cat2 in self.categories[cat1].get('incompatible', []):
                return False
        
        if cat2 in self.categories:
            if cat1 in self.categories[cat2].get('incompatible', []):
                return False
        
        return True
