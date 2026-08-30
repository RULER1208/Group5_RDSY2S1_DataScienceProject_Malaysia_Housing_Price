"""
BMDS2003 Data Science - Deployment Prototype
Malaysia Housing Median Price Estimator

Run locally:  streamlit run my_UI2_location_persistence_jinjang_fixed.py
"""
# Build: figures-exact-map-poi-fixed
from __future__ import annotations
from pathlib import Path
from html import escape
from difflib import SequenceMatcher
import inspect
import joblib
import pandas as pd
import streamlit as st
import hashlib
import re

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    px = None
    go = None
    HAS_PLOTLY = False

try:
    import folium
    from folium.plugins import MarkerCluster
    from streamlit_folium import st_folium
    HAS_INTERACTIVE_MAP = True
except ImportError:
    folium = None
    MarkerCluster = None
    st_folium = None
    HAS_INTERACTIVE_MAP = False

try:
    from geopy.geocoders import Nominatim
    HAS_GEOPY = True
except ImportError:
    HAS_GEOPY = False

from area_preprocessing import clean_area_name, create_area_key, display_name

st.set_page_config(
    page_title="Malaysia Housing Price Estimator",
    page_icon=":material/home:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CONFIGURATION & REAL-WORLD COORDINATES
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "malaysia_house_price_cleaned_with_area.csv"
RESULTS_PATH = APP_DIR / "model_comparison_table.csv"
RESULTS_FALLBACK_PATH = APP_DIR / "model_results.csv"
MODELS_DIR = APP_DIR / "models"
FIGURES_DIR = APP_DIR / "figures"
MODEL_FEATURES = ["State", "Area_Key", "Tenure", "Primary_Type", "Median_PSF", "Transactions"]

NO_AREA = "— No Area Selected —"

STATE_COORDS = {
    "Johor": [1.9344, 103.3587], "Kedah": [6.1184, 100.3685],
    "Kelantan": [5.3500, 102.0000], "Melaka": [2.2500, 102.2500],
    "Negeri Sembilan": [2.7258, 101.9424], "Pahang": [3.8126, 102.8000],
    "Penang": [5.4141, 100.3288], "Perak": [4.5921, 101.0901],
    "Perlis": [6.4449, 100.2048], "Sabah": [5.4204, 116.7968],
    "Sarawak": [2.5574, 113.0012], "Selangor": [3.0738, 101.5183],
    "Terengganu": [4.7500, 103.0000], "Kuala Lumpur": [3.1390, 101.6869],
    "Putrajaya": [2.9264, 101.6964], "Labuan": [5.2831, 115.2308]
}


# ---------------------------------------------------------------------------
# MAP COVERAGE
# ---------------------------------------------------------------------------
# Nationwide map coverage combines three layers:
# 1) every State -> Area pair in the uploaded 2025 housing dataset,
# 2) all 160 DOSM administrative-district units across Malaysia, and
# 3) useful official/local subdivisions for the four DOSM state-level units
#    that are not split into lower administrative districts in that dataset.
#
# The prediction model still receives the selected location directly. Its
# OneHotEncoder was trained with handle_unknown="infrequent_if_exist", so
# official locations absent from the housing data remain valid inputs but are
# identified in the result as out-of-dataset area estimates.

OFFICIAL_DISTRICTS = {'Johor': ['Batu Pahat',
           'Johor Bahru',
           'Kluang',
           'Kota Tinggi',
           'Kulai',
           'Mersing',
           'Muar',
           'Pontian',
           'Segamat',
           'Tangkak'],
 'Kedah': ['Baling',
           'Bandar Baharu',
           'Kota Setar',
           'Kuala Muda',
           'Kubang Pasu',
           'Kulim',
           'Langkawi',
           'Padang Terap',
           'Pendang',
           'Pokok Sena',
           'Sik',
           'Yan'],
 'Kelantan': ['Bachok',
              'Kota Bharu',
              'Machang',
              'Pasir Mas',
              'Pasir Puteh',
              'Tanah Merah',
              'Tumpat',
              'Gua Musang',
              'Kuala Krai',
              'Jeli',
              'Kecil Lojing'],
 'Melaka': ['Alor Gajah', 'Jasin', 'Melaka Tengah'],
 'Negeri Sembilan': ['Jelebu', 'Jempol', 'Kuala Pilah', 'Port Dickson', 'Rembau', 'Seremban', 'Tampin'],
 'Pahang': ['Bentong',
            'Bera',
            'Cameron Highlands',
            'Jerantut',
            'Kuantan',
            'Lipis',
            'Maran',
            'Pekan',
            'Raub',
            'Rompin',
            'Temerloh'],
 'Penang': ['Barat Daya',
            'Seberang Perai Selatan',
            'Seberang Perai Tengah',
            'Seberang Perai Utara',
            'Timur Laut'],
 'Perak': ['Batang Padang',
           'Manjung',
           'Kinta',
           'Kerian',
           'Kuala Kangsar',
           'Larut & Matang',
           'Hilir Perak',
           'Hulu Perak',
           'Perak Tengah',
           'Kampar',
           'Muallim',
           'Bagan Datuk',
           'Selama'],
 'Perlis': ['Perlis'],
 'Sabah': ['Tawau',
           'Lahad Datu',
           'Semporna',
           'Sandakan',
           'Kinabatangan',
           'Beluran',
           'Kota Kinabalu',
           'Ranau',
           'Kota Belud',
           'Tuaran',
           'Penampang',
           'Papar',
           'Kudat',
           'Kota Marudu',
           'Pitas',
           'Beaufort',
           'Kuala Penyu',
           'Sipitang',
           'Tenom',
           'Nabawan',
           'Keningau',
           'Tambunan',
           'Kunak',
           'Tongod',
           'Putatan',
           'Telupid',
           'Kalabakan'],
 'Sarawak': ['Kuching',
             'Bau',
             'Lundu',
             'Samarahan',
             'Serian',
             'Simunjan',
             'Sri Aman',
             'Lubok Antu',
             'Betong',
             'Saratok',
             'Sarikei',
             'Maradong',
             'Daro',
             'Julau',
             'Sibu',
             'Dalat',
             'Mukah',
             'Kanowit',
             'Bintulu',
             'Tatau',
             'Kapit',
             'Song',
             'Belaga',
             'Miri',
             'Marudi',
             'Limbang',
             'Lawas',
             'Matu',
             'Asajaya',
             'Pakan',
             'Selangau',
             'Tebedu',
             'Pusa',
             'Kabong',
             'Tanjung Manis',
             'Sebauh',
             'Bukit Mabong',
             'Subis',
             'Beluru',
             'Telang Usan'],
 'Selangor': ['Gombak',
              'Hulu Langat',
              'Hulu Selangor',
              'Klang',
              'Kuala Langat',
              'Kuala Selangor',
              'Petaling',
              'Sabak Bernam',
              'Sepang'],
 'Terengganu': ['Besut',
                'Dungun',
                'Hulu Terengganu',
                'Kemaman',
                'Kuala Nerus',
                'Kuala Terengganu',
                'Marang',
                'Setiu'],
 'Kuala Lumpur': ['Kuala Lumpur'],
 'Labuan': ['Labuan'],
 'Putrajaya': ['Putrajaya']}

assert len(OFFICIAL_DISTRICTS) == 16
assert sum(len(v) for v in OFFICIAL_DISTRICTS.values()) == 160

SPECIAL_DISPLAY_AREAS = {'Putrajaya': ['Precinct 1',
               'Precinct 2',
               'Precinct 3',
               'Precinct 4',
               'Precinct 5',
               'Precinct 6',
               'Precinct 7',
               'Precinct 8',
               'Precinct 9',
               'Precinct 10',
               'Precinct 11',
               'Precinct 12',
               'Precinct 13',
               'Precinct 14',
               'Precinct 15',
               'Precinct 16',
               'Precinct 17',
               'Precinct 18',
               'Precinct 19',
               'Precinct 20'],
 'Perlis': ['Titi Tinggi',
            'Beseri',
            'Chuping',
            'Paya',
            'Padang Siding',
            'Abi',
            'Padang Pauh',
            'Ngulang',
            'Oran',
            'Kurong Batang',
            'Arau',
            'Kechor',
            'Sena',
            'Sungai Adam',
            'Kurong Anai',
            'Jejawi',
            'Kuala Perlis',
            'Wang Bintong',
            'Seriab',
            'Kayang',
            'Utan Aji',
            'Sanglang'],
 'Kuala Lumpur': ['City Centre',
                  'Wangsa Maju - Maluri',
                  'Bukit Jalil - Seputeh',
                  'Bandar Tun Razak - Sungai Besi',
                  'Sentul - Menjalara',
                  'Damansara - Penchala'],
 'Labuan': ['Batu Arang',
            'Batu Manikar',
            'Bebuloh',
            'Belukut',
            'Bukit Kallam',
            'Bukit Kuda',
            'Durian Tunjong',
            'Ganggarak',
            'Gersik/Saguking',
            'Kerupang/Nagalang',
            'Kilan/Pulau Akar',
            'Lajau',
            'Layang-Layangan',
            'Lubuk Temiang',
            'Pantai',
            'Patau-Patau 1',
            'Patau-Patau 2',
            'Pohon Batu',
            'Rancha-Rancha',
            'Sungai Bangat',
            'Sungai Bedaun',
            'Sungai Buton',
            'Sungai Keling',
            'Sungai Labu',
            'Sungai Lada',
            'Sungai Miri',
            'Tanjung Aru']}

SPECIAL_AREA_LABELS = {'Putrajaya': 'Precinct', 'Perlis': 'Mukim', 'Kuala Lumpur': 'Strategic zone', 'Labuan': 'Village area'}


# Dataset validation snapshot (from malaysia_house_price_data_2025.csv).
# This is NOT loaded at runtime and does not control the UI design. It is only
# merged into the nationwide map list so every State -> Area pair found in the
# supplied dataset is guaranteed to remain available on the map.
VALIDATED_DATASET_AREAS = {'Johor': ['Bakri',
           'Batu Pahat',
           'Gelang Patah',
           'Gerisek',
           'Iskandar Puteri (Nusajaya)',
           'Jementah',
           'Johor Bahru',
           'Kluang',
           'Kota Tinggi',
           'Kulai',
           'Labis',
           'Masai',
           'Mersing',
           'Muar',
           'Pagoh',
           'Paloh',
           'Pasir Gudang',
           'Pengerang',
           'Perling',
           'Permas Jaya',
           'Pontian',
           'Segamat',
           'Senai',
           'Senggarang',
           'Simpang Rengam',
           'Skudai',
           'Tampoi',
           'Tangkak',
           'Tebrau',
           'Ulu Tiram',
           'Yong Peng'],
 'Kedah': ['Alor Setar',
           'Bandar Baharu',
           'Bedong',
           'Gurun',
           'Jitra',
           'Kota Sarang Semut',
           'Kuah',
           'Kuala Kedah',
           'Kuala Ketil',
           'Kulim',
           'Lunas',
           'Merbok',
           'Padang Serai',
           'Pokok Sena',
           'Sungai Lalang',
           'Sungai Petani',
           'Teloi Kiri'],
 'Kelantan': ['Cherang Ruku', 'Machang', 'Padang Enggang'],
 'Kuala Lumpur': ['Ampang',
                  'Ampang Hilir',
                  'Bandar Menjalara',
                  'Bandar Tasik Selatan',
                  'Bangsar',
                  'Batu Caves',
                  'Brickfields',
                  'Bukit Bintang',
                  'Bukit Jalil',
                  'Cheras',
                  'City Centre',
                  'Damansara Heights',
                  'Desa ParkCity',
                  'Desa Petaling',
                  'Dutamas',
                  'Jalan Ipoh',
                  'Jalan Klang Lama (Old Klang Road)',
                  'Jalan Kuching',
                  'Jinjang',
                  'KL City',
                  'KL Sentral',
                  'KLCC',
                  'Kampung Kerinchi (Bangsar South)',
                  'Kepong',
                  'Kuchai Lama',
                  'Mont Kiara',
                  'Pantai',
                  'Salak Selatan',
                  'Segambut',
                  'Sentul',
                  'Setapak',
                  'Setiawangsa',
                  'Sri Hartamas',
                  'Sri Petaling',
                  'Sungai Besi',
                  'Taman Desa',
                  'Taman Tun Dr Ismail',
                  'Wangsa Maju'],
 'Labuan': ['Labuan'],
 'Melaka': ['Alor Gajah',
            'Ayer Molek',
            'Bachang',
            'Balai Panjang',
            'Batu Berendam',
            'Bemban',
            'Bertam',
            'Bukit Baru',
            'Bukit Katil',
            'Bukit Rambai',
            'Cheng',
            'Durian Tunggal',
            'Duyong',
            'Jasin',
            'Klebang',
            'Krubong',
            'Kuala Sungai Baru',
            'Masjid Tanah',
            'Melaka City',
            'Merlimau',
            'Paya Rumput',
            'Sungai Rambai',
            'Sungai Udang',
            'Sungei Baru Tengah',
            'Sungei Petai',
            'Tanjong Kling',
            'Tanjong Minyak',
            'Umbai'],
 'Negeri Sembilan': ['Ampangan',
                     'Bahau',
                     'Bandar Enstek',
                     'Bandar Sri Sendayan',
                     'Bukit Kepayang',
                     'Gemas',
                     'Jimah',
                     'Juasseh',
                     'Kuala Pilah',
                     'Labu',
                     'Lenggeng',
                     'Linggi',
                     'Lukut',
                     'Mantin',
                     'Nilai',
                     'Paroi',
                     'Pasir Panjang',
                     'Port Dickson',
                     'Rantau',
                     'Rasah',
                     'Rembau',
                     'Senawang',
                     'Seremban',
                     'Seremban 2',
                     'Sikamat',
                     'Simpang Pertang',
                     'Tampin',
                     'Telok Kemang'],
 'Pahang': ['Bentong',
            'Chenor',
            'Genting Highlands',
            'Hulu Lepar',
            'Kuala Lipis',
            'Kuantan',
            'Mentakab',
            'Pekan',
            'Raub',
            'Rompin',
            'Sungai Karang',
            'Teras',
            'Triang'],
 'Penang': ['Ayer Itam',
            'Balik Pulau',
            'Batu Ferringhi',
            'Batu Kawan',
            'Bayan Baru',
            'Bayan Lepas',
            'Bukit Jambul',
            'Bukit Mertajam',
            'Bukit Minyak',
            'Butterworth',
            'Gelugor',
            'Georgetown',
            'Gurney',
            'Jelutong',
            'Juru',
            'Kepala Batas',
            'Kubang Semang',
            'Nibong Tebal',
            'Penaga',
            'Perai',
            'Permatang Pauh',
            'Relau',
            'Seberang Jaya',
            'Simpang Ampat',
            'Sungai Ara',
            'Sungai Bakap',
            'Sungai Dua',
            'Sungai Jawi',
            'Tanjong Tokong',
            'Tanjung Bungah',
            'Tasek Gelugor',
            'Teluk Kumbar'],
 'Perak': ['Bagan Serai',
           'Batu Gajah',
           'Batu Kurau',
           'Bidor',
           'Chemor',
           'Chenderiang',
           'Gerik',
           'Gopeng',
           'Hutan Melintang',
           'Ipoh',
           'Kampar',
           'Kamunting',
           'Kuala Kangsar',
           'Lahat',
           'Lumut',
           'Menglembu',
           'Padang Rengas',
           'Parit Buntar',
           'Pengkalan Hulu',
           'Pusing',
           'Seri Iskandar',
           'Seri Manjong',
           'Simpang',
           'Simpang Pulai',
           'Sitiawan',
           'Sungai Siput',
           'Taiping',
           'Tambun',
           'Tanjong Tualang',
           'Tapah',
           'Teluk Intan',
           'Tronoh',
           'Ulu Bernam'],
 'Perlis': ['Arau'],
 'Putrajaya': ['Putrajaya'],
 'Sabah': ['Kota Kinabalu', 'Kota Marudu', 'Lahad Datu', 'Papar', 'Penampang', 'Sandakan', 'Tawau', 'Tuaran'],
 'Sarawak': ['Bintulu', 'Kota Samarahan', 'Kuching', 'Limbang', 'Miri', 'Sibu', 'Sri Aman'],
 'Selangor': ['Ampang',
              'Ara Damansara',
              'Balakong',
              'Bandar Kinrara',
              'Bandar Puncak Alam',
              'Bandar Sri Damansara',
              'Bandar Sungai Long',
              'Bandar Sunway',
              'Bandar Utama',
              'Bangi',
              'Banting',
              'Batang Kali',
              'Batu Arang',
              'Batu Caves',
              'Beranang',
              'Cheras',
              'Cyberjaya',
              'Damansara Damai',
              'Damansara Perdana',
              'Dengkil',
              'Glenmarie',
              'Hulu Langat',
              'Hulu Selangor',
              'Ijok',
              'Jenjarom',
              'Kajang',
              'Kapar',
              'Kelana Jaya',
              'Kepong',
              'Klang',
              'Kota Damansara',
              'Kuala Kubu Baru',
              'Kuala Selangor',
              'Mutiara Damansara',
              'Pandamaran',
              'Petaling Jaya',
              'Port Klang',
              'Puchong',
              'Rasa',
              'Rawang',
              'Sabak Bernam',
              'Saujana',
              'Saujana Utama',
              'Selayang',
              'Semenyih',
              'Sepang',
              'Serendah',
              'Seri Kembangan',
              'Setia Alam',
              'Shah Alam',
              'Subang Jaya',
              'Sungai Besar',
              'Sungai Buloh',
              'Tanjong Duabelas',
              'Telok Panglima Garang',
              'Tropicana',
              'Ulu Klang',
              'Ulu Langat'],
 'Terengganu': ['Besut', 'Dungun', 'Hulu Terengganu', 'Kemaman', 'Kerteh', 'Kijal', 'Kuala Ibai', 'Kuala Terengganu']}


# Massively expanded pre-calculated real-world coordinates for exact map pins
HARDCODED_AREAS = {
    "Selangor": {
        "Sekinchan": [3.5053, 101.1036], "Tanjong Karang": [3.4267, 101.1773],
        "Pandamaran": [3.0132, 101.4172], "Kuala Selangor": [3.3364, 101.2504],
        "Sabak Bernam": [3.7667, 100.9833], "Sungai Besar": [3.6833, 100.9833],
        "Banting": [2.8155, 101.4975], "Petaling Jaya": [3.1073, 101.6067], 
        "Shah Alam": [3.0738, 101.5183], "Subang Jaya": [3.0471, 101.5832],
        "Klang": [3.0449, 101.4456], "Puchong": [3.0246, 101.6168], 
        "Kajang": [2.9935, 101.7892], "Cheras": [3.1062, 101.7690],
        "Rawang": [3.3213, 101.5822], "Cyberjaya": [2.9228, 101.6572],
        "Setia Alam": [3.1110, 101.4450], "Bukit Beruntung": [3.3100, 101.5540],
        "Bandar Saujana Putra": [2.9490, 101.5790], "Semenyih": [2.9480, 101.8440],
        "Bangi": [2.9200, 101.7800], "Serdang": [3.0220, 101.7100],
        "Batu Caves": [3.2380, 101.6810], "Ampang": [3.1490, 101.7610],
        "Sungai Buloh": [3.2080, 101.5790], "Gombak": [3.2200, 101.7000],
        "Sepang": [2.6865, 101.7483], "Selayang": [3.2505, 101.6448],
    },
    "Kuala Lumpur": {
        "Bukit Bintang": [3.1460, 101.7110], "Setapak": [3.1895, 101.7058], 
        "Kepong": [3.2120, 101.6358], "Mont Kiara": [3.1672, 101.6508], 
        "Bukit Jalil": [3.0578, 101.6885], "Wangsa Maju": [3.2045, 101.7348], 
        "Bangsar": [3.1253, 101.6749], "Old Klang Road": [3.0830, 101.6740],
        "Jinjang": [3.21131, 101.65832],
    },
    "Johor": {
        "Skudai": [1.5333, 103.6667], "Tebrau": [1.5833, 103.7500], 
        "Pasir Gudang": [1.4703, 103.8966], "Kulai": [1.6561, 103.6023], 
        "Johor Bahru": [1.4927, 103.7414], "Batu Pahat": [1.8548, 102.9325],
        "Kluang": [2.0251, 103.3328], "Muar": [2.0442, 102.5689], 
        "Pontian": [1.4883, 103.3888], "Kota Tinggi": [1.7381, 103.8999],
        "Segamat": [2.5144, 102.8159], "Mersing": [2.4312, 103.8361],
    },
    "Perak": {
        "Tapah": [4.2000, 101.2600], "Ipoh": [4.5975, 101.0901],
        "Taiping": [4.8500, 100.7333], "Teluk Intan": [4.0259, 101.0213],
        "Sitiawan": [4.2144, 100.6974], "Seri Manjong": [4.1950, 100.6650], 
        "Kampar": [4.3000, 101.1500], "Lumut": [4.2333, 100.6333],
        "Chenderiang": [4.2667, 101.2333],
    },
    "Penang": {
        "Georgetown": [5.4141, 100.3288], "Butterworth": [5.3995, 100.3638], 
        "Bayan Lepas": [5.2952, 100.2588], "Tasek Gelugor": [5.4833, 100.4833],
        "Bukit Mertajam": [5.3629, 100.4666], "Perai": [5.3833, 100.3833],
        "Batu Kawan": [5.2652, 100.4283], "Nibong Tebal": [5.1667, 100.4667],
        "Kepala Batas": [5.5167, 100.4333],
    },
    "Melaka": {
        "Bemban": [2.2667, 102.3667], "Jasin": [2.3130, 102.4312],
        "Ayer Keroh": [2.2642, 102.2858], "Alor Gajah": [2.3833, 102.2000],
    },
    "Negeri Sembilan": {
        "Seremban": [2.7297, 101.9381], "Port Dickson": [2.5228, 101.7959],
        "Nilai": [2.8167, 101.8000],
    },
    "Kedah": {
        "Alor Setar": [6.1210, 100.3601], "Sungai Petani": [5.6436, 100.4897],
        "Kulim": [5.3667, 100.5500],
    },
    "Pahang": {
        "Kuantan": [3.8077, 103.3260], "Temerloh": [3.4506, 102.4168], 
        "Cameron Highlands": [4.4721, 101.3801]
    },
    "Kelantan": { "Kota Bharu": [6.1254, 102.2381] },
    "Terengganu": { "Kuala Terengganu": [5.3302, 103.1408], "Kemaman": [4.2333, 103.3333] },
    "Sabah": { "Kota Kinabalu": [5.9804, 116.0735] },
    "Sarawak": { "Kuching": [1.5533, 110.3592] }
}

# ---------------------------------------------------------------------------
# SVG ICONS & LIGHT DESIGN SYSTEM
# ---------------------------------------------------------------------------
SVG_PATHS = {
    "home": '<path d="m3 11 9-7 9 7"/><path d="M5 10v10h14V10"/><path d="M9 20v-6h6v6"/>',
    "location": '<path d="M20 10c0 5-8 11-8 11S4 15 4 10a8 8 0 1 1 16 0Z"/><circle cx="12" cy="10" r="2.5"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/>',
    "pin": '<path d="M12 22s7-6 7-13a7 7 0 1 0-14 0c0 7 7 13 7 13Z"/><circle cx="12" cy="9" r="2"/>',
    "building": '<path d="M4 21V7l8-4 8 4v14"/><path d="M8 9h2m4 0h2M8 13h2m4 0h2M8 17h2m4 0h2M2 21h20"/>',
    "key": '<circle cx="8" cy="15" r="4"/><path d="m11 12 9-9m-4 4 3 3m-7 1 3 3"/>',
    "money": '<rect x="3" y="6" width="18" height="12" rx="2"/><path d="M7 10h.01M17 14h.01"/><circle cx="12" cy="12" r="2.5"/>',
    "chart": '<path d="M4 19V9m6 10V5m6 14v-7m4 7H2"/>',
    "model": '<rect x="5" y="5" width="14" height="14" rx="3"/><path d="M9 9h6v6H9zM9 2v3m6-3v3M9 19v3m6-3v3M2 9h3m14 0h3M2 15h3m14 0h3"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10h.01"/>',
    "warning": '<path d="M12 3 2 21h20L12 3Z"/><path d="M12 9v5m0 3h.01"/>',
    "refresh": '<path d="M20 6v5h-5M4 18v-5h5"/><path d="M18 11a7 7 0 0 0-12-4L4 11m2 2a7 7 0 0 0 12 4l2-4"/>',
    "filter": '<path d="M3 5h18M6 12h12m-8 7h4"/>',
    "arrow": '<path d="M5 12h14m-5-5 5 5-5 5"/>',
    "save": '<path d="M5 3h12l2 2v16H5z"/><path d="M8 3v6h8V3M8 21v-7h8v7"/>',
}


def svg_icon(name: str, size: int = 20, color: str = "currentColor") -> str:
    """Return an accessible decorative inline SVG from the app icon set."""
    paths = SVG_PATHS.get(name, SVG_PATHS["info"])
    return (
        f'<svg aria-hidden="true" focusable="false" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" '
        f'stroke-linecap="round" stroke-linejoin="round">{paths}</svg>'
    )


def section_header(step: int, title: str, note: str, icon: str) -> None:
    st.markdown(
        f'<div class="mh-section-head"><div class="mh-step">{step}</div>'
        f'<div class="mh-section-icon">{svg_icon(icon, 21)}</div><div>'
        f'<h2 class="mh-section-title">{escape(title)}</h2>'
        f'<p class="mh-section-note">{escape(note)}</p></div></div>',
        unsafe_allow_html=True,
    )


def metric_cards(items) -> None:
    cards = "".join(
        f'<div class="mh-metric"><div class="k">{escape(str(label))}</div>'
        f'<div class="v">{escape(str(value))}</div>'
        f'<div class="hint">{escape(str(hint))}</div></div>'
        for label, value, hint in items
    )
    st.markdown(f'<div class="mh-metric-row">{cards}</div>', unsafe_allow_html=True)


st.markdown(r"""
<style>
:root {
    --ink:#1E293B;
    --muted:#667085;
    --line:#DDE3EC;
    --surface:#FFFFFF;
    --page:#F6F8FC;
    --blue:#4F6FEA;
    --blue-dark:#3F5CC7;
    --blue-soft:#EEF2FF;
    --green:#2F8F68;
    --green-dark:#247653;
    --green-soft:#EAF7F0;
    --green-pale:#F4FBF7;
    --teal:#0D9488;
    --teal-soft:#CCFBF1;
    --amber:#F59E0B;
    --amber-soft:#FFFBEB;
    --red:#EF4444;
    --mono:ui-monospace,"SF Mono",monospace;
    --shadow:0 12px 30px rgba(23,35,59,.10);
    --radius:20px;
}

.stApp { background:var(--page); color:var(--ink); }
.block-container { max-width:1240px; padding-top:14px!important; padding-bottom:2.2rem; }
header[data-testid="stHeader"] { background:transparent!important; }
div[data-testid="stToolbar"] { display:none!important; }
#MainMenu { visibility:hidden; }
footer { visibility:hidden; }

/* ---------- LIGHT UI HEADER / NAVIGATION ---------- */
.stTabs [role="tablist"] {
    display:flex!important;
    align-items:center!important;
    gap:8px!important;
    min-height:82px;
    padding:14px 18px!important;
    background:#FFFFFF!important;
    border:1px solid var(--line)!important;
    border-radius:21px!important;
    margin:10px 0 28px!important;
    box-shadow:0 10px 24px rgba(23,35,59,.06);
}
.stTabs [role="tablist"]::before {
    content:"⌂  Malaysia Housing Estimator";
    white-space:nowrap;
    display:flex;
    align-items:center;
    min-width:320px;
    height:42px;
    padding:0 22px 0 10px;
    margin-right:14px;
    border-right:1px solid var(--line);
    font-size:1.08rem;
    font-weight:800;
    letter-spacing:.01em;
    color:var(--ink);
}
.stTabs [role="tab"] {
    align-self:stretch!important;
    height:auto!important;
    padding:0 24px!important;
    border-radius:12px!important;
    color:var(--muted)!important;
    font-weight:650!important;
    font-size:.94rem!important;
    background:transparent!important;
    border:1px solid transparent!important;
    transition:all .16s ease!important;
    cursor:pointer!important;
}
.stTabs [role="tab"]:hover { color:var(--ink)!important; background:#F1F5F9!important; }
.stTabs [role="tab"][aria-selected="true"] {
    color:var(--blue-dark)!important;
    background:var(--blue-soft)!important;
    border-color:#C7D2FE!important;
    box-shadow:0 4px 10px rgba(79,111,234,.14)!important;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none!important; }
.stTabs [data-baseweb="tab-panel"] { min-height:60vh; }

/* ---------- HERO / REPORT PANELS USED ON OTHER TABS ---------- */
.mh-hero { display:flex; justify-content:space-between; align-items:center; gap:24px;
    padding:26px 28px; margin-bottom:20px; background:linear-gradient(135deg,#FFF,#EFF6FF);
    border:1px solid #DBEAFE; border-radius:24px; box-shadow:var(--shadow); }
.mh-hero h1 { margin:0; font-size:clamp(1.65rem,3vw,2.35rem); color:var(--ink); }
.mh-hero p { margin:.55rem 0 0; color:var(--muted); max-width:700px; line-height:1.55; }
.mh-hero-icon { width:64px; height:64px; flex:0 0 64px; display:grid; place-items:center;
    border-radius:18px; color:#2563EB; background:#DBEAFE; }

/* ---------- OLD UI SECTION HEADERS ---------- */
.mh-section-head { display:flex; align-items:flex-start; gap:13px; margin:2px 0 16px; }
.mh-step {
    width:38px; height:38px; flex:0 0 38px; border-radius:12px;
    background:var(--blue-soft); color:var(--blue-dark);
    display:flex; align-items:center; justify-content:center;
    font-family:var(--mono); font-weight:800; border:1px solid #D9E0F7;
}
.mh-section-icon { display:none; }
.mh-section-title { margin:0!important; color:#1E293B; font-size:1.38rem; line-height:1.2; }
.mh-section-note { color:var(--muted); font-size:.94rem; margin:.3rem 0 0; }
.mh-label { font-family:var(--mono); font-size:.75rem; letter-spacing:.16em; color:#667085; margin:6px 0 8px; text-transform:uppercase; }
.mh-rule { border:none; border-top:1px solid var(--line); margin:26px 0 22px; }
.mh-help { color:var(--muted); font-size:.82rem; line-height:1.45; margin-top:6px; }

/* ---------- MAP / LOCATION ---------- */
.mh-map-wrap { border:1px solid var(--line); border-radius:20px; overflow:hidden;
    box-shadow:0 10px 26px rgba(23,35,59,.08); background:#FFFFFF; }
.mh-chiprow { display:flex; flex-wrap:wrap; gap:10px; margin-top:10px; }
.mh-chip { display:inline-flex; align-items:center; gap:8px; background:#FFFFFF; border:1px solid #DDE3EC;
    padding:10px 15px; min-height:48px; border-radius:14px; color:#475569; font-size:.95rem;
    box-shadow:0 3px 10px rgba(23,35,59,.04); }
.mh-chip strong { color:#1E293B; }
.mh-map-legend { display:flex; align-items:center; flex-wrap:wrap; gap:10px 18px; margin-top:10px;
    color:#667085; font-size:.82rem; }
.mh-map-legend .legend-title { font-weight:700; color:#475467; margin-right:2px; }
.mh-map-legend .legend-item { display:inline-flex; align-items:center; gap:7px; white-space:nowrap; }
.mh-map-legend .legend-dot { width:12px; height:12px; border-radius:50%; display:inline-block; box-sizing:border-box; }
.mh-map-legend .state-dot { background:#2F6FED; border:2px solid #93C5FD; }
.mh-map-legend .verified-dot { background:#F59E0B; border:2px solid #C47A10; }
.mh-map-legend .approx-dot { background:#D0D5DD; border:2px solid #98A2B3; }
.mh-map-legend .selected-dot { background:#10B981; border:2px solid #18875D; }

/* ---------- OLD UI PROPERTY BUTTONS ---------- */
div[class*="st-key-btn_"] button[kind="secondary"],
div[class*="st-key-btn_"] button[kind="primary"] {
    min-height:82px!important; border-radius:17px!important; white-space:pre-line!important;
    line-height:1.24!important; padding:9px 5px!important; font-weight:650!important;
    font-size:.76rem!important; text-align:center!important; transition:all .16s ease!important;
}
div[class*="st-key-btn_"] button[kind="secondary"] { background:#FFFFFF!important; color:#334155!important;
    border:1px solid #DDE3EC!important; box-shadow:0 5px 14px rgba(23,35,59,.05)!important; }
div[class*="st-key-btn_"] button[kind="secondary"]:hover { background:#F8FAFC!important; border-color:#B8C5D8!important;
    color:#243A5A!important; transform:translateY(-1px); }
div[class*="st-key-btn_"] button[kind="primary"] { background:var(--green-soft)!important; color:var(--green-dark)!important;
    border:1.5px solid #76BE9E!important; box-shadow:0 7px 16px rgba(47,143,104,.12)!important; }
div[class*="st-key-btn_"] button p { margin:0!important; white-space:pre-line!important; overflow-wrap:anywhere!important; }
.mh-property-card { min-height:96px; text-align:center; padding:13px 7px 7px; border:1px solid var(--line);
    border-radius:16px 16px 5px 5px; background:#FFF; color:#475569; transition:.16s ease; }
.mh-property-card:hover { transform:translateY(-1px); border-color:#93C5FD; }
.mh-property-card.selected { background:#EFF6FF; border:2px solid #2563EB; color:#1D4ED8; }
.mh-property-card .icon { margin:auto; width:34px; height:34px; display:grid; place-items:center; border-radius:10px; background:#F1F5F9; }
.mh-property-card.selected .icon { background:#DBEAFE; }
.mh-property-card .label { margin-top:8px; font-size:.78rem; font-weight:800; line-height:1.2; }

/* ---------- TENURE ---------- */
.tenure-hint { color:#8A94A6; font-size:.78rem; margin:-2px 0 7px; }
div[class*="st-key-tenure_btn_"] button { width:100%!important; min-height:40px!important; padding:6px 12px!important;
    border-radius:999px!important; font-size:.86rem!important; font-weight:650!important; line-height:1.1!important;
    transition:all .15s ease!important; box-shadow:none!important; }
div[class*="st-key-tenure_btn_"] button[kind="secondary"] { background:#FFFFFF!important; color:#5A6475!important; border:1px solid #D9E0EA!important; }
div[class*="st-key-tenure_btn_"] button[kind="secondary"]:hover { background:#F8FAFC!important; border-color:#BCC7D6!important; color:#263852!important; }
div[class*="st-key-tenure_btn_"] button[kind="primary"] { background:var(--green-soft)!important; color:var(--green-dark)!important;
    border:1.5px solid #76BE9E!important; box-shadow:0 3px 9px rgba(47,143,104,.10)!important; }
div[class*="st-key-tenure_btn_"] button p { margin:0!important; white-space:nowrap!important; }

/* ---------- INPUTS / BUTTONS ---------- */
div[data-testid="stNumberInputContainer"], div[data-baseweb="input"] { background:#FFFFFF; border:1px solid #DDE3EC; border-radius:14px; }
.stButton>button[kind="primary"] { background:linear-gradient(180deg, #5876E8 0%, var(--blue) 100%);
    border-color:var(--blue); border-radius:14px; min-height:50px; font-weight:750; box-shadow:0 10px 20px rgba(79,111,234,.18); }
.stButton>button[kind="primary"]:hover { background:var(--blue-dark); border-color:var(--blue-dark); }
button[kind="secondary"] { border-radius:14px!important; }

/* ---------- LIGHT MICRO-INTERACTIONS ---------- */
div[class*="st-key-btn_"] button {
    transition:transform .16s cubic-bezier(.22,1,.36,1), box-shadow .16s ease,
        border-color .16s ease, background-color .16s ease!important;
}
div[class*="st-key-btn_"] button:hover { transform:translateY(-2px); }
div[class*="st-key-btn_"] button[kind="primary"] {
    transform:translateY(-2px);
    box-shadow:0 8px 18px rgba(47,143,104,.14)!important;
}
div[class*="st-key-btn_"] button:active { transform:translateY(0) scale(.985); }

div[class*="st-key-tenure_btn_"] button {
    transition:transform .15s cubic-bezier(.22,1,.36,1), box-shadow .15s ease,
        border-color .15s ease, background-color .15s ease!important;
}
div[class*="st-key-tenure_btn_"] button:hover { transform:translateY(-1px); }
div[class*="st-key-tenure_btn_"] button[kind="primary"] {
    transform:translateY(-1px);
    box-shadow:0 5px 12px rgba(47,143,104,.12)!important;
}
div[class*="st-key-tenure_btn_"] button:active { transform:scale(.985); }

div.st-key-generate_estimate button {
    transition:transform .16s cubic-bezier(.22,1,.36,1), box-shadow .16s ease,
        background-color .16s ease!important;
}
div.st-key-generate_estimate button:hover {
    transform:translateY(-2px);
    box-shadow:0 13px 24px rgba(79,111,234,.23)!important;
}
div.st-key-generate_estimate button:active { transform:translateY(0) scale(.995); }

@media (prefers-reduced-motion:reduce) {
    div[class*="st-key-btn_"] button,
    div[class*="st-key-tenure_btn_"] button,
    div.st-key-generate_estimate button {
        transition:none!important; transform:none!important;
    }
}

/* ---------- NEW UI RESULT CARD KEPT ---------- */
.mh-result { margin-top:22px; padding:28px; background:linear-gradient(135deg,#FFFFFF,#F0FDFA);
    border:1px solid #99F6E4; border-left:5px solid var(--teal); border-radius:24px; box-shadow:var(--shadow); }
.mh-result-top { display:flex; justify-content:space-between; gap:20px; align-items:flex-start; }
.mh-result .cap { color:var(--teal); font-size:.73rem; font-weight:850; letter-spacing:.1em; text-transform:uppercase; }
.mh-result .price { color:#0F172A; font-size:clamp(2.35rem,6vw,3.65rem); line-height:1; font-weight:850; margin:11px 0 9px; }
.mh-result .sub { color:#475569; line-height:1.5; }
.mh-range { display:inline-block; margin-top:13px; padding:8px 12px; border-radius:10px; color:#0F766E; background:#CCFBF1; font-weight:800; }
.mh-result-badge { min-width:210px; padding:14px 16px; background:#FFFFFF; border:1px solid var(--line); border-radius:15px; }
.mh-result-badge .kicker { color:var(--muted); font-size:.68rem; letter-spacing:.09em; text-transform:uppercase; }
.mh-result-badge .model { color:#0F172A; font-size:1.04rem; font-weight:800; margin-top:5px; }
.mh-stats { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:20px; padding-top:18px; border-top:1px solid var(--line); }
.mh-stat { padding:13px 14px; background:#FFFFFF; border:1px solid var(--line); border-radius:14px; }
.mh-stat .k { color:var(--muted); font-size:.68rem; font-weight:800; letter-spacing:.07em; text-transform:uppercase; }
.mh-stat .v { color:#0F172A; font-size:1rem; font-weight:850; margin-top:5px; }
.mh-warning { display:flex; gap:10px; margin-top:14px; padding:13px 15px; color:#92400E;
    background:var(--amber-soft); border:1px solid #FDE68A; border-radius:13px; font-size:.86rem; line-height:1.5; }
.mh-disclaimer { color:var(--muted); margin-top:13px; font-size:.82rem; }
.mh-empty { padding:38px 24px; margin-top:20px; text-align:center; color:var(--muted); background:#FFFFFF;
    border:1px dashed #CBD5E1; border-radius:20px; box-shadow:0 5px 18px rgba(23,35,59,.05); }
.mh-empty .icon { width:42px; height:42px; margin:0 auto 10px; display:grid; place-items:center;
    border-radius:13px; color:var(--blue); background:#EFF6FF; }

/* ---------- INSIGHTS / REPORT ---------- */
.mh-panel, .mh-filter-card { background:#FFFFFF; border:1px solid var(--line); border-radius:20px;
    padding:20px 22px; box-shadow:0 8px 22px rgba(23,35,59,.06); margin-bottom:18px; }
.mh-panel-title { display:flex; align-items:center; gap:9px; color:var(--ink); font-weight:800; margin-bottom:5px; }
.mh-metric-row { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:12px 0 20px; }
.mh-metric { background:#FFFFFF; border-radius:16px;
    padding:15px 17px; border:1px solid var(--line); box-shadow:0 6px 16px rgba(23,35,59,.05); }
.mh-metric .k { font-family:var(--mono); font-size:.65rem; letter-spacing:.12em; color:var(--muted); text-transform:uppercase; }
.mh-metric .v { font-family:var(--mono); font-size:1.28rem; font-weight:800; color:var(--ink); margin-top:6px; }
.mh-metric .hint { color:var(--muted); font-size:.73rem; margin-top:4px; }
.mh-figure-card { margin:15px 0 8px; padding:16px 18px 8px; background:#FFFFFF; border:1px solid var(--line); border-radius:18px; }
.mh-figure-title { display:flex; align-items:center; gap:8px; color:var(--ink); font-weight:800; }
.mh-figure-insight { color:#475569; margin:7px 0 4px; font-size:.9rem; }
.mh-why { color:var(--muted); font-size:.82rem; margin-bottom:2px; }
.mh-takeaway { padding:15px 17px; margin:14px 0 20px; background:#EFF6FF; border:1px solid #BFDBFE; border-radius:15px; color:#1E3A8A; }
.mh-model-hero { padding:24px; margin-bottom:18px; background:linear-gradient(135deg,#FFF,#EFF6FF);
    border:1px solid #BFDBFE; border-radius:22px; box-shadow:var(--shadow); }
.mh-model-name { color:var(--ink); font-size:1.75rem; font-weight:850; margin:4px 0 8px; }
.mh-badge { display:inline-block; padding:5px 9px; border-radius:999px; background:#DBEAFE; color:#1D4ED8; font-size:.72rem; font-weight:850; }
.mh-limitations { padding:18px 20px; margin-top:20px; background:#FFFBEB; border:1px solid #FDE68A; border-radius:17px; color:#78350F; }
.mh-footer { margin-top:36px; padding:18px 22px; border-radius:18px;
    background:#FFFFFF; border:1px solid var(--line);
    color:var(--muted); display:flex; align-items:center; justify-content:space-between; gap:18px;
    box-shadow:0 8px 22px rgba(23,35,59,.06); border-top:3px solid var(--blue); }
.mh-footer .brand { color:var(--ink); font-weight:800; letter-spacing:.01em; }
.mh-footer .sub { color:var(--muted); font-size:.84rem; }

@media (max-width:900px) {
    .stTabs [role="tablist"] { flex-wrap:wrap!important; }
    .stTabs [role="tablist"]::before { width:100%; min-width:0; }
    .mh-stats, .mh-metric-row { grid-template-columns:1fr 1fr; }
    .mh-chip { width:100%; justify-content:center; }
    .mh-result-top, .mh-footer { flex-direction:column; align-items:flex-start; }
    .mh-result-badge { width:100%; min-width:auto; }
}
@media (max-width:600px) {
    .block-container { padding-left:1rem!important; padding-right:1rem!important; }
    .mh-stats, .mh-metric-row { grid-template-columns:1fr; }
    .mh-hero { padding:20px; }
    .mh-hero-icon { display:none; }
    .stTabs [role="tab"] { padding:0 12px!important; font-size:.82rem!important; }
}

/* ---------- FINAL LIGHT-THEME SAFETY OVERRIDE ---------- */
/* Keep this block last in the stylesheet: it re-asserts the light theme
   with !important so no leftover/cached dark-navy rule (top nav, metric
   cards, footer, map-legend borders) can win the cascade. */
.stTabs [role="tablist"], .mh-metric, .mh-footer { background-image:none!important; }
.stTabs [role="tablist"] { background-color:#FFFFFF!important; border-color:var(--line)!important; }
.stTabs [role="tablist"]::before { color:var(--ink)!important; border-right-color:var(--line)!important; }
.stTabs [role="tab"] { color:var(--muted)!important; }
.stTabs [role="tab"][aria-selected="true"] { background-color:var(--blue-soft)!important; color:var(--blue-dark)!important; }
.mh-metric { background-color:#FFFFFF!important; border-color:var(--line)!important; }
.mh-metric .k, .mh-metric .hint { color:var(--muted)!important; }
.mh-metric .v { color:var(--ink)!important; }
.mh-footer { background-color:#FFFFFF!important; border-color:var(--line)!important; color:var(--muted)!important; }
.mh-footer .brand { color:var(--ink)!important; }
.mh-footer .sub { color:var(--muted)!important; }
.mh-map-legend .state-dot { border-color:#93C5FD!important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# POSTCODE -> STATE LOOKUP (offline, deterministic — no network dependency)
# ---------------------------------------------------------------------------
# Standard Malaysian postcode prefix ranges (first two digits). Used as the
# primary signal for address/postcode detection so the feature keeps working
# even when there is no internet access or the geocoding service is
# unreachable/rate-limited.
POSTCODE_STATE_RANGES = [
    (1, 2, "Perlis"),
    (5, 9, "Kedah"),
    (10, 14, "Penang"),
    (15, 18, "Kelantan"),
    (20, 24, "Terengganu"),
    (25, 28, "Pahang"),
    (30, 36, "Perak"),
    (39, 39, "Pahang"),          # Cameron Highlands
    (40, 48, "Selangor"),
    (49, 49, "Selangor"),
    (50, 60, "Kuala Lumpur"),
    (62, 62, "Putrajaya"),
    (63, 64, "Selangor"),        # Cyberjaya / Sepang
    (68, 68, "Kuala Lumpur"),
    (70, 73, "Negeri Sembilan"),
    (75, 78, "Melaka"),
    (79, 86, "Johor"),
    (87, 87, "Labuan"),
    (88, 91, "Sabah"),
    (93, 98, "Sarawak"),
]


def get_state_from_postcode(postcode: str):
    """Deterministic offline lookup — always responds, regardless of network."""
    if not postcode or len(postcode) != 5 or not postcode.isdigit():
        return None
    prefix = int(postcode[:2])
    for lo, hi, state in POSTCODE_STATE_RANGES:
        if lo <= prefix <= hi:
            return state
    return None

# ---------------------------------------------------------------------------
# DATA & MAP HELPERS
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_PATH)

RESULT_COLUMNS = [
    "Model",
    "Group_CV_RMSE_mean",
    "Group_CV_RMSE_std",
    "Group_CV_MAE_mean",
    "Group_CV_R2_mean",
    "RMSE_train",
    "RMSE_test",
    "MAE_test",
    "MAPE_test_pct",
    "R2_test",
    "Selected_Model",
]


def resolve_results_path() -> Path:
    if RESULTS_PATH.exists():
        return RESULTS_PATH
    if RESULTS_FALLBACK_PATH.exists():
        return RESULTS_FALLBACK_PATH
    return RESULTS_PATH


@st.cache_data(show_spinner=False)
def load_results():
    results = pd.read_csv(resolve_results_path())
    results = results.loc[:, ~results.columns.str.startswith("Unnamed:")].copy()

    missing = [column for column in RESULT_COLUMNS if column not in results.columns]
    if missing:
        raise ValueError(
            "The model results file is missing required columns: "
            + ", ".join(missing)
        )

    return (
        results[RESULT_COLUMNS]
        .sort_values(["Group_CV_RMSE_mean", "Group_CV_RMSE_std"])
        .reset_index(drop=True)
    )


def selected_model_mask(results: pd.DataFrame) -> pd.Series:
    return (
        results["Selected_Model"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def selected_model_name(results: pd.DataFrame) -> str:
    selected = results.loc[selected_model_mask(results), "Model"]
    if len(selected):
        return str(selected.iloc[0])

    best_mean = results["Group_CV_RMSE_mean"].min()
    practical_ties = results[
        results["Group_CV_RMSE_mean"] <= best_mean * 1.01
    ]
    return str(
        practical_ties.sort_values(
            ["Group_CV_RMSE_std", "Group_CV_RMSE_mean"]
        ).iloc[0]["Model"]
    )


@st.cache_resource(show_spinner=False)
def load_model(name):
    filename = name.split(" (")[0].lower().replace(" ", "_") + ".pkl"
    return joblib.load(MODELS_DIR / filename)


def model_has_seen_area(model, area_key: str) -> bool:
    try:
        categorical_pipeline = model.named_steps["preprocess"].named_transformers_["cat"]
        encoder = categorical_pipeline.named_steps["encoder"]
        feature_names = list(getattr(encoder, "feature_names_in_", []))
        if "Area_Key" in feature_names:
            area_position = feature_names.index("Area_Key")
        else:
            area_position = 1
        return area_key in set(encoder.categories_[area_position])
    except Exception:
        return False

def get_area_coords(area_name: str, state_name: str):
    """Return a verified precomputed coordinate without network calls."""
    state_name = str(state_name or "").strip()
    area_name = str(area_name or "").strip()

    state_areas = HARDCODED_AREAS.get(state_name, {})

    if area_name in state_areas:
        return list(state_areas[area_name])

    wanted = clean_area_name(area_name).lower()

    for known_area, coords in state_areas.items():
        if clean_area_name(known_area).lower() == wanted:
            return list(coords)

    if clean_area_name(area_name).lower() == clean_area_name(state_name).lower():
        coords = STATE_COORDS.get(state_name)
        return list(coords) if coords else None

    return None


def get_area_map_coords(area_name: str, state_name: str):
    """Return the best available coordinate for an area."""
    real = get_area_coords(area_name, state_name)

    if real:
        return real, False

    base = STATE_COORDS.get(state_name)

    if not base:
        return None, True

    h = int(
        hashlib.md5(
            f"{state_name}|{area_name}".encode("utf-8")
        ).hexdigest(),
        16,
    )

    lat_offset = (((h % 1000) / 999) - 0.5) * 0.55
    lon_offset = ((((h // 1000) % 1000) / 999) - 0.5) * 0.75

    return [
        base[0] + lat_offset,
        base[1] + lon_offset,
    ], True

def field_label(text: str) -> None:
    st.markdown(f'<div class="mh-label">{text}</div>', unsafe_allow_html=True)

def get_property_icon(ptype: str) -> str:
    ptype_icons = {
        "Bungalow": "🏠", "Semi D": "🏘️", "Cluster House": "🏡",
        "Terrace House": "🏘️", "Town House": "🏠", "Condominium": "🏢",
        "Service Residence": "🏙️", "Apartment": "🏢", "Flat": "🏬",
    }
    return ptype_icons.get(str(ptype), "🏠")


def get_property_label(ptype: str) -> str:
    return f"{get_property_icon(ptype)}\n{ptype}"


def get_area_marker_data(state_name: str):
    """Build the marker table using the latest coordinates."""
    marker_rows = []

    for area_name in get_areas_for_state(state_name):
        coordinates, approximate = get_area_map_coords(
            area_name,
            state_name,
        )

        if coordinates:
            marker_rows.append(
                (
                    area_name,
                    coordinates[0],
                    coordinates[1],
                    approximate,
                    get_area_source(state_name, area_name),
                )
            )

    return marker_rows

@st.cache_data(show_spinner=False)
def get_dataset_areas_for_state(state_name: str):
    return list(VALIDATED_DATASET_AREAS.get(state_name, []))

@st.cache_data(show_spinner=False)
def get_areas_for_state(state_name: str):
    # Static nationwide list: official districts + official/local subdivisions
    # + any State -> Area pairs found in the supplied housing dataset.
    areas = set(OFFICIAL_DISTRICTS.get(state_name, []))
    areas.update(SPECIAL_DISPLAY_AREAS.get(state_name, []))
    areas.update(VALIDATED_DATASET_AREAS.get(state_name, []))
    return sorted(areas)

def get_area_source(state_name: str, area_name: str) -> str:
    dataset_areas = set(VALIDATED_DATASET_AREAS.get(state_name, []))
    in_dataset = area_name in dataset_areas
    in_districts = area_name in set(OFFICIAL_DISTRICTS.get(state_name, []))
    in_special = area_name in set(SPECIAL_DISPLAY_AREAS.get(state_name, []))

    if in_dataset and in_districts:
        return "Dataset locality · Official district"
    if in_dataset:
        return "Housing dataset locality"
    if in_districts:
        return "Official administrative district"
    if in_special:
        return SPECIAL_AREA_LABELS.get(state_name, "Official local area")
    return "Malaysia map area"

def is_dataset_area(state_name: str, area_name: str) -> bool:
    return area_name in set(VALIDATED_DATASET_AREAS.get(state_name, []))

@st.cache_data(show_spinner=False)
def map_coverage_summary():
    dataset_pairs = {
        (state_name, area_name)
        for state_name, areas in VALIDATED_DATASET_AREAS.items()
        for area_name in areas
    }
    nationwide_pairs = {
        (state_name, area_name)
        for state_name in STATE_COORDS
        for area_name in get_areas_for_state(state_name)
    }
    return {
        "states": sorted(STATE_COORDS),
        "state_count": len(STATE_COORDS),
        "dataset_pairs": len(dataset_pairs),
        "official_district_units": int(sum(len(v) for v in OFFICIAL_DISTRICTS.values())),
        "nationwide_state_area_pairs": len(nationwide_pairs),
        "dataset_pairs_missing_from_map": len(dataset_pairs - nationwide_pairs),
    }


# ---------------------------------------------------------------------------
# FAST MALAYSIA-WIDE LOCATION SEARCH HELPERS
# ---------------------------------------------------------------------------
STATE_ALIASES = {
    "kl": "Kuala Lumpur", "k l": "Kuala Lumpur", "wpkl": "Kuala Lumpur",
    "w p kuala lumpur": "Kuala Lumpur", "wilayah persekutuan kuala lumpur": "Kuala Lumpur",
    "kuala lumpur": "Kuala Lumpur", "selangor": "Selangor", "johor": "Johor",
    "johor bahru": "Johor", "penang": "Penang", "pulau pinang": "Penang",
    "melaka": "Melaka", "malacca": "Melaka", "negeri sembilan": "Negeri Sembilan",
    "ns": "Negeri Sembilan", "pahang": "Pahang", "perak": "Perak", "perlis": "Perlis",
    "kedah": "Kedah", "kelantan": "Kelantan", "terengganu": "Terengganu",
    "sabah": "Sabah", "sarawak": "Sarawak", "putrajaya": "Putrajaya", "labuan": "Labuan",
}

AREA_ALIASES = {
    ("Kuala Lumpur", "Jalan Klang Lama (Old Klang Road)"): ["old klang road", "jalan klang lama", "okr"],
    ("Kuala Lumpur", "Kampung Kerinchi (Bangsar South)"): ["bangsar south", "kampung kerinchi"],
    ("Kuala Lumpur", "KLCC"): ["klcc", "kuala lumpur city centre"],
    ("Kuala Lumpur", "KL City"): ["kl city", "kuala lumpur city"],
    ("Johor", "Iskandar Puteri (Nusajaya)"): ["iskandar puteri", "nusajaya"],
    ("Penang", "Georgetown"): ["george town", "georgetown"],
    ("Selangor", "Petaling Jaya"): ["pj", "petaling jaya"],
    ("Selangor", "Subang Jaya"): ["subang jaya", "usj"],
    ("Selangor", "Bandar Sunway"): ["sunway", "bandar sunway"],
}


def normalise_lookup_text(value: str) -> str:
    value = str(value or "").lower()
    value = value.replace("w.p.", "wilayah persekutuan")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def contains_lookup_phrase(haystack: str, needle: str) -> bool:
    haystack = normalise_lookup_text(haystack)
    needle = normalise_lookup_text(needle)
    if not haystack or not needle:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack) is not None


@st.cache_data(show_spinner=False)
def build_area_lookup_index():
    rows = []
    for state_name in sorted(STATE_COORDS):
        for area_name in get_areas_for_state(state_name):
            variants = {area_name, clean_area_name(area_name), display_name(clean_area_name(area_name))}
            variants.update(AREA_ALIASES.get((state_name, area_name), []))
            for variant in variants:
                norm = normalise_lookup_text(variant)
                if norm:
                    rows.append({
                        "state": state_name,
                        "area": area_name,
                        "variant": norm,
                        "score": len(norm) + (8 if area_name in VALIDATED_DATASET_AREAS.get(state_name, []) else 0),
                    })
    rows.sort(key=lambda item: (item["score"], len(item["variant"])), reverse=True)
    return rows


def match_state_text(addr_norm: str):
    matches = []
    for alias, state_name in STATE_ALIASES.items():
        alias_norm = normalise_lookup_text(alias)
        if contains_lookup_phrase(addr_norm, alias_norm):
            matches.append((len(alias_norm), state_name))
    if matches:
        return sorted(matches, reverse=True)[0][1]
    return None


def match_area_text(addr_norm: str, preferred_states=None):
    preferred_states = [state for state in (preferred_states or []) if state]
    matches = []
    for row in build_area_lookup_index():
        if contains_lookup_phrase(addr_norm, row["variant"]):
            preference_bonus = 0
            if row["state"] in preferred_states:
                preference_bonus = 45 - preferred_states.index(row["state"])
            matches.append((row["score"] + preference_bonus, row["state"], row["area"]))
    if matches:
        _, state_name, area_name = sorted(matches, reverse=True)[0]
        return state_name, area_name
    return None, None


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 14)
def geocode_malaysia_candidates(query: str):
    """Return several Malaysia-only Nominatim candidates instead of trusting the first hit.

    The old UI accepted exactly_one=True, so nonsense text such as ``xyz`` could be
    mapped to an unrelated place.  This function deliberately returns a shortlist;
    the caller then validates how strongly each result matches the user's words.
    """
    if not HAS_GEOPY or not query:
        return []
    try:
        geolocator = Nominatim(
            user_agent="malaysia_housing_estimator_student",
            timeout=3,
        )
        locations = geolocator.geocode(
            f"{query}, Malaysia",
            country_codes="my",
            addressdetails=True,
            namedetails=True,
            exactly_one=False,
            limit=6,
        )
        if not locations:
            return []
        if not isinstance(locations, (list, tuple)):
            locations = [locations]

        candidates = []
        for loc in locations:
            raw = getattr(loc, "raw", {}) or {}
            details = raw.get("address", {}) or {}
            namedetails = raw.get("namedetails", {}) or {}
            display_text = raw.get("display_name") or getattr(loc, "address", "") or query
            primary_name = (
                namedetails.get("name")
                or details.get("amenity")
                or details.get("building")
                or details.get("tourism")
                or str(display_text).split(",", 1)[0]
            )
            candidates.append({
                "lat": float(loc.latitude),
                "lon": float(loc.longitude),
                "display_name": str(display_text),
                "primary_name": str(primary_name or ""),
                "state_raw": details.get("state") or details.get("region") or "",
                "area_raw": (
                    details.get("suburb") or details.get("quarter") or details.get("neighbourhood") or
                    details.get("village") or details.get("town") or details.get("city_district") or
                    details.get("city") or details.get("municipality") or details.get("county") or
                    details.get("district") or ""
                ),
                "postcode": details.get("postcode") or "",
                "category": raw.get("category") or raw.get("class") or "",
                "type": raw.get("type") or "",
                "importance": float(raw.get("importance") or 0.0),
                "namedetails": {str(k): str(v) for k, v in namedetails.items() if v},
            })
        return candidates
    except Exception:
        return []


def _compact_lookup_text(value: str) -> str:
    """Normalised text without spaces, useful for acronyms such as TAR UMT/TARUMT."""
    return normalise_lookup_text(value).replace(" ", "")


def geocode_query_score(query: str, candidate: dict) -> float:
    """Estimate whether a Nominatim result genuinely resembles the user's query.

    This is intentionally conservative.  A geocoder result is not accepted merely
    because it exists in Malaysia; the searched words must also appear in, or closely
    resemble, the returned POI/place name.  This prevents ``xyz -> Sibu`` style jumps.
    """
    q = normalise_lookup_text(query)
    q_compact = _compact_lookup_text(query)
    if not q or not q_compact:
        return 0.0

    name = normalise_lookup_text(candidate.get("primary_name", ""))
    display = normalise_lookup_text(candidate.get("display_name", ""))
    named_values = [normalise_lookup_text(v) for v in (candidate.get("namedetails") or {}).values()]
    texts = [t for t in [name, display, *named_values] if t]
    if not texts:
        return 0.0

    compact_texts = [t.replace(" ", "") for t in texts]
    exact_phrase = any(contains_lookup_phrase(t, q) for t in texts)
    compact_exact = len(q_compact) >= 4 and any(q_compact in t for t in compact_texts)

    q_tokens = [token for token in q.split() if len(token) >= 2]
    candidate_tokens = set(" ".join(texts).split())
    token_coverage = (
        sum(1 for token in q_tokens if token in candidate_tokens) / len(q_tokens)
        if q_tokens else 0.0
    )

    # Compare primarily with the returned POI/place name, then the whole display label.
    name_ratio = SequenceMatcher(None, q_compact, name.replace(" ", "")).ratio() if name else 0.0
    display_head = display.split(" ")[:8]
    display_ratio = SequenceMatcher(None, q_compact, "".join(display_head)).ratio() if display_head else 0.0
    fuzzy = max(name_ratio, display_ratio)

    score = 0.0
    if exact_phrase:
        score += 100.0
    elif compact_exact:
        score += 96.0
    else:
        score += 52.0 * token_coverage
        score += 42.0 * fuzzy

    # A little preference for Nominatim's own ranking, but never enough to rescue
    # a semantically unrelated candidate.
    score += min(6.0, max(0.0, float(candidate.get("importance", 0.0))) * 10.0)

    # Very short arbitrary strings are especially dangerous unless they literally
    # occur in the returned place name/display label.
    if len(q_compact) <= 3 and not exact_phrase:
        score = min(score, 35.0)

    return round(score, 3)


def validated_geocode_candidates(query: str):
    """Return only candidates that are strongly supported by the typed query."""
    # Three-character free-text searches are too ambiguous for POI geocoding.
    # Useful Malaysian abbreviations such as KL/PJ/USJ/KLCC are already handled
    # by the deterministic state/area alias lookup before this function runs.
    if len(_compact_lookup_text(query)) <= 3:
        return []

    ranked = []
    seen = set()
    for candidate in geocode_malaysia_candidates(query):
        score = geocode_query_score(query, candidate)
        if score < 72.0:
            continue
        key = (
            round(float(candidate["lat"]), 5),
            round(float(candidate["lon"]), 5),
            normalise_lookup_text(candidate.get("display_name", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        item = dict(candidate)
        item["query_score"] = score
        ranked.append(item)
    ranked.sort(key=lambda item: (item["query_score"], item.get("importance", 0.0)), reverse=True)
    return ranked[:5]


def resolve_candidate_state(candidate: dict, available_states):
    """Resolve state from explicit metadata, address text, or postcode fallback."""
    forced = candidate.get("forced_state")
    if forced in available_states:
        return forced

    text_sources = [
        candidate.get("state_raw", ""),
        candidate.get("display_name", ""),
        candidate.get("area_raw", ""),
        candidate.get("primary_name", ""),
    ]
    for source in text_sources:
        source_norm = normalise_lookup_text(source)
        if not source_norm:
            continue
        for state_name in available_states:
            state_norm = normalise_lookup_text(state_name)
            if (
                contains_lookup_phrase(source_norm, state_norm)
                or contains_lookup_phrase(state_norm, source_norm)
            ):
                return state_name
        alias_state = match_state_text(source_norm)
        if alias_state in available_states:
            return alias_state

    postcode = re.search(r"\b\d{5}\b", str(candidate.get("postcode") or candidate.get("display_name") or ""))
    if postcode:
        postcode_state = get_state_from_postcode(postcode.group())
        if postcode_state in available_states:
            return postcode_state
    return None


def resolve_candidate_area(candidate: dict, state_name: str | None):
    """Translate a POI/address into the map/model area vocabulary when possible."""
    forced = str(candidate.get("forced_area") or "").strip()
    if forced:
        return forced

    preferred = [state_name] if state_name else []
    # Try structured locality first, then the full display label.  The latter is
    # important for results such as "Chung Kwok ... Sentul, Kuala Lumpur" where
    # Nominatim may put Chow Kit or another neighbourhood in area_raw.
    for source in [candidate.get("area_raw", ""), candidate.get("display_name", "")]:
        source_norm = normalise_lookup_text(source)
        if not source_norm:
            continue
        area_state, area_name = match_area_text(source_norm, preferred_states=preferred)
        if area_state and area_name and (not state_name or area_state == state_name):
            return area_name

    raw_area = str(candidate.get("area_raw", "") or "").strip()
    if raw_area:
        return display_name(clean_area_name(raw_area))
    return None


def candidate_label(candidate: dict) -> str:
    primary = str(candidate.get("primary_name") or "").strip()
    display_text = str(candidate.get("display_name") or "").strip()
    if candidate.get("source") == "known_branch":
        return display_text or primary
    if primary and primary.lower() not in display_text.lower():
        return f"{primary} — {display_text}"
    return display_text or primary or "Possible Malaysia location"



# ---------------------------------------------------------------------------
# WELL-KNOWN MULTI-BRANCH POI SEARCHES
# ---------------------------------------------------------------------------
# Some organisations have several Malaysian branches but a generic Nominatim
# search may only return one of them.  Keep a small explicit branch registry for
# these cases so the user can choose the intended campus before the map moves.
# TAR UMT branch names/addresses follow the university's official campus list.
TAR_UMT_BRANCHES = [
    {
        "primary_name": "TAR UMT Kuala Lumpur Campus",
        "display_name": "TAR UMT Kuala Lumpur Campus — Jalan Genting Kelang, Setapak, 53300 Kuala Lumpur",
        "forced_state": "Kuala Lumpur",
        "forced_area": "Setapak",
        "lat": 3.2137,
        "lon": 101.7263,
        "source": "known_branch",
    },
    {
        "primary_name": "TAR UMT Penang Branch",
        "display_name": "TAR UMT Penang Branch — Lorong Lembah Permai Tiga, Tanjong Bungah, Penang",
        "forced_state": "Penang",
        "forced_area": "Tanjung Bungah",
        "lat": 5.4660,
        "lon": 100.2910,
        "source": "known_branch",
    },
    {
        "primary_name": "TAR UMT Perak Branch",
        "display_name": "TAR UMT Perak Branch — Jalan Kolej, Taman Bandar Baru, Kampar, Perak",
        "forced_state": "Perak",
        "forced_area": "Kampar",
        "lat": 4.3300,
        "lon": 101.1420,
        "source": "known_branch",
    },
    {
        "primary_name": "TAR UMT Johor Branch",
        "display_name": "TAR UMT Johor Branch — Jalan Segamat / Labis, Segamat, Johor",
        "forced_state": "Johor",
        "forced_area": "Segamat",
        "lat": 2.5144,
        "lon": 102.8159,
        "source": "known_branch",
    },
    {
        "primary_name": "TAR UMT Pahang Branch",
        "display_name": "TAR UMT Pahang Branch — Indera Mahkota 9, Kuantan, Pahang",
        "forced_state": "Pahang",
        "forced_area": "Kuantan",
        "lat": 3.8077,
        "lon": 103.3260,
        "source": "known_branch",
    },
    {
        "primary_name": "TAR UMT Sabah Branch",
        "display_name": "TAR UMT Sabah Branch — Jalan Alamesra, Alamesra, Kota Kinabalu, Sabah",
        "forced_state": "Sabah",
        "forced_area": "Kota Kinabalu",
        "lat": 5.9804,
        "lon": 116.0735,
        "source": "known_branch",
    },
]


def special_multi_branch_candidates(query: str):
    """Return a complete branch list for recognised multi-campus organisations."""
    q = normalise_lookup_text(query)
    compact = _compact_lookup_text(query)
    tarumt_aliases = {
        "tarumt", "taruc", "tarc",
        "tunkuabdulrahmanuniversityofmanagementandtechnology",
        "tunkuabdulrahmanuniversitycollege",
    }
    phrase_match = (
        "tunku abdul rahman university of management and technology" in q
        or "tunku abdul rahman university college" in q
    )
    if compact in tarumt_aliases or phrase_match:
        result = []
        for idx, branch in enumerate(TAR_UMT_BRANCHES):
            item = dict(branch)
            item.update({
                "query_score": 100.0,
                "importance": 1.0,
                "state_raw": branch["forced_state"],
                "area_raw": branch["forced_area"],
                "namedetails": {"name": branch["primary_name"]},
                "candidate_id": f"tarumt_{idx}",
            })
            result.append(item)
        return result
    return []

def model_default_name(results: pd.DataFrame) -> str:
    model_options = results["Model"].astype(str).tolist()
    for model_name in model_options:
        if normalise_lookup_text(model_name) == "random forest":
            return model_name
    for model_name in model_options:
        if "random" in normalise_lookup_text(model_name) and "forest" in normalise_lookup_text(model_name):
            return model_name
    return selected_model_name(results)

# ---------------------------------------------------------------------------
# LOGIC CONTROLLERS
# ---------------------------------------------------------------------------
def reset_location_state():
    st.session_state["selected_state"] = None
    st.session_state["selected_area"] = None
    st.session_state["map_center"] = [4.2105, 108.9758]
    st.session_state["map_zoom"] = 6
    st.session_state["address_input"] = ""
    st.session_state["address_feedback"] = None
    st.session_state["last_prediction"] = None
    st.session_state["last_map_popup"] = None
    st.session_state["searched_point"] = None
    st.session_state["map_fly_request"] = None
    st.session_state["location_candidates"] = []
    st.session_state["location_candidate_query"] = ""
    st.session_state.pop("location_candidate_choice", None)


def _apply_location_match(
    *,
    matched_state,
    matched_area,
    searched_point,
    previous_map_center,
    previous_map_zoom,
    feedback_message,
):
    """Commit one validated location and prepare the one-time Spider-Man/map flight."""
    st.session_state["last_prediction"] = None
    st.session_state["selected_state"] = matched_state
    st.session_state["selected_area"] = matched_area
    st.session_state["searched_point"] = searched_point
    st.session_state["location_candidates"] = []
    st.session_state["location_candidate_query"] = ""
    st.session_state.pop("location_candidate_choice", None)

    target_center = STATE_COORDS.get(matched_state, [4.2105, 108.9758])
    target_zoom = 8

    if matched_area:
        if searched_point:
            target_center = [searched_point["lat"], searched_point["lon"]]
            target_zoom = 14
        else:
            coords, _ = get_area_map_coords(matched_area, matched_state)
            if coords:
                target_center = list(coords)
                target_zoom = 12
    elif searched_point:
        target_center = [searched_point["lat"], searched_point["lon"]]
        target_zoom = 13

    st.session_state["address_feedback"] = feedback_message
    st.session_state["map_center"] = list(target_center)
    st.session_state["map_zoom"] = int(target_zoom)
    st.session_state["map_fly_request"] = {
        "from_center": list(previous_map_center),
        "from_zoom": int(previous_map_zoom),
        "to_center": list(target_center),
        "to_zoom": int(target_zoom),
        "duration": 3.4,
    }


def apply_geocode_candidate(candidate: dict, available_states):
    """Apply a validated candidate selected by the user."""
    previous_map_center = list(st.session_state.get("map_center", [4.2105, 108.9758]))
    previous_map_zoom = int(st.session_state.get("map_zoom", 6))

    matched_state = resolve_candidate_state(candidate, available_states)
    if not matched_state:
        st.session_state["address_feedback"] = (
            "warning",
            "This search result could not be linked to a Malaysian state. Please choose another result or select the location on the map.",
        )
        return False

    matched_area = resolve_candidate_area(candidate, matched_state)
    searched_point = {
        "lat": float(candidate["lat"]),
        "lon": float(candidate["lon"]),
        "label": candidate.get("display_name") or candidate.get("primary_name") or "Searched location",
    }

    place_name = str(candidate.get("primary_name") or "").strip()
    if matched_area:
        if place_name and normalise_lookup_text(place_name) != normalise_lookup_text(matched_area):
            message = (
                "success",
                f"Found **{place_name}**. Housing area used: **{matched_area}, {matched_state}**.",
            )
        else:
            message = ("success", f"Matched **{matched_area}, {matched_state}**.")
    else:
        message = ("info", f"Matched state **{matched_state}**. Pick an area on the map below.")

    _apply_location_match(
        matched_state=matched_state,
        matched_area=matched_area,
        searched_point=searched_point,
        previous_map_center=previous_map_center,
        previous_map_zoom=previous_map_zoom,
        feedback_message=message,
    )
    return True


def analyze_address(available_states):
    addr_raw = st.session_state.get("address_input", "")
    addr_norm = normalise_lookup_text(addr_raw)
    if not addr_norm:
        st.session_state["address_feedback"] = None
        st.session_state["location_candidates"] = []
        return

    # Starting a new search invalidates any older ambiguity shortlist.
    st.session_state["location_candidates"] = []
    st.session_state["location_candidate_query"] = ""

    previous_map_center = list(st.session_state.get("map_center", [4.2105, 108.9758]))
    previous_map_zoom = int(st.session_state.get("map_zoom", 6))

    # 1) Trust deterministic local evidence first: a known area, state, or postcode.
    postcode_match = re.search(r"\b\d{5}\b", str(addr_raw))
    postcode_state = get_state_from_postcode(postcode_match.group()) if postcode_match else None
    text_state = match_state_text(addr_norm)
    preferred_states = [state for state in [postcode_state, text_state] if state in available_states]
    area_state, area_name = match_area_text(addr_norm, preferred_states=preferred_states)

    if area_state and area_name:
        area_coords, _ = get_area_map_coords(
            area_name,
            area_state,
        )

        direct_area_point = None

        if area_coords:
            direct_area_point = {
                "lat": float(area_coords[0]),
                "lon": float(area_coords[1]),
                "label": f"{area_name}, {area_state}",
                "area_reference": True,
            }

        _apply_location_match(
            matched_state=area_state,
            matched_area=area_name,
            searched_point=direct_area_point,
            previous_map_center=previous_map_center,
            previous_map_zoom=previous_map_zoom,
            feedback_message=(
                "success",
                f"Matched **{area_name}, {area_state}**.",
            ),
        )
        return

    # A pure/clear state name is deterministic.  A postcode also gives a reliable
    # state, but not necessarily an exact area, so the user can choose the area map pin.
    if postcode_state or text_state:
        matched_state = postcode_state or text_state
        _apply_location_match(
            matched_state=matched_state,
            matched_area=None,
            searched_point=None,
            previous_map_center=previous_map_center,
            previous_map_zoom=previous_map_zoom,
            feedback_message=("info", f"Matched state **{matched_state}**. Pick an area on the map below."),
        )
        return

    # 2) Known multi-branch organisations are handled before general geocoding.
    # A generic search for TAR UMT often returns only the Penang branch, so a
    # search for "tarumt" intentionally shows all six official campuses.
    branch_candidates = special_multi_branch_candidates(addr_raw)
    if branch_candidates:
        st.session_state["location_candidates"] = branch_candidates
        st.session_state["location_candidate_query"] = str(addr_raw)
        st.session_state["location_candidate_choice"] = branch_candidates[0]["candidate_id"]
        st.session_state["address_feedback"] = (
            "info",
            "TAR UMT has multiple Malaysian campuses. Please choose the campus you mean before the map moves.",
        )
        return

    # 3) POI/building/free-text search: get several Malaysia candidates, then reject
    # results whose returned name does not actually resemble what the user typed.
    candidates = validated_geocode_candidates(addr_raw) if HAS_GEOPY else []
    if not candidates:
        st.session_state["searched_point"] = None
        st.session_state["address_feedback"] = (
            "warning",
            f"Couldn't confidently recognise **{escape(str(addr_raw))}**. Try an area, postcode, building name, or select the location on the map.",
        )
        return

    # If several candidates are nearly tied, do not silently choose one.  This is
    # useful for organisations with multiple campuses/branches and similarly named POIs.
    top_score = float(candidates[0]["query_score"])
    plausible = [c for c in candidates if top_score - float(c["query_score"]) <= 8.0]
    distinct = []
    seen_places = set()
    for candidate in plausible:
        state = resolve_candidate_state(candidate, available_states)
        key = (
            state,
            normalise_lookup_text(candidate.get("area_raw", "")),
            round(float(candidate["lat"]), 3),
            round(float(candidate["lon"]), 3),
        )
        if key not in seen_places:
            seen_places.add(key)
            distinct.append(candidate)

    if len(distinct) > 1:
        shortlist = []
        for idx, candidate in enumerate(distinct[:5]):
            item = dict(candidate)
            item["candidate_id"] = item.get("candidate_id") or f"geo_{idx}_{round(float(item['lat']), 5)}_{round(float(item['lon']), 5)}"
            shortlist.append(item)
        st.session_state["location_candidates"] = shortlist
        st.session_state["location_candidate_query"] = str(addr_raw)
        st.session_state["location_candidate_choice"] = shortlist[0]["candidate_id"]
        st.session_state["address_feedback"] = (
            "info",
            "Several believable locations were found. Please choose the correct one below before the map moves.",
        )
        return

    apply_geocode_candidate(candidates[0], available_states)

# ---------------------------------------------------------------------------
# PAGE 1 - PREDICTION INTERFACE
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# PAGE 1 - PREDICTION INTERFACE
# ---------------------------------------------------------------------------
def clear_last_prediction():
    st.session_state["last_prediction"] = None


def market_reference(data, state, area, ptype, tenure):
    state_mask = data["State"].eq(state) if state else pd.Series(False, index=data.index)
    type_mask = data["Primary_Type"].eq(ptype)
    tenure_mask = data["Tenure"].eq(tenure)
    candidates = []
    if state and area:
        area_mask = data["Area_Clean"].eq(clean_area_name(area))
        candidates.extend([
            ("selected area, property type and tenure", state_mask & area_mask & type_mask & tenure_mask),
            ("selected area and property type", state_mask & area_mask & type_mask),
            ("selected area", state_mask & area_mask),
        ])
    if state:
        candidates.extend([
            ("selected state, property type and tenure", state_mask & type_mask & tenure_mask),
            ("selected state and property type", state_mask & type_mask),
            ("selected state", state_mask),
        ])
    candidates.append(("full 2025 dataset", pd.Series(True, index=data.index)))
    for label, mask in candidates:
        pool = data.loc[mask]
        if len(pool):
            return {"label": label, "rows": len(pool), "psf": int(round(pool["Median_PSF"].median())),
                    "transactions": int(round(pool["Transactions"].median()))}
    return {"label": "full 2025 dataset", "rows": len(data),
            "psf": int(round(data["Median_PSF"].median())),
            "transactions": int(round(data["Transactions"].median()))}


def render_stepper(current_state, current_area, prediction):
    active = 3 if prediction else 2 if current_state and current_area else 1
    items = [
        (1, "Select Location", "State and area"),
        (2, "Choose Property", "Type, tenure and PSF"),
        (3, "Generate Estimate", "Review your result"),
    ]
    html = "".join(
        f'<div class="mh-step-card {"active" if number == active else ""}">'
        f'<div class="mh-step-num">{number}</div><div class="mh-step-text">'
        f'<strong>{label}</strong><span>{detail}</span></div></div>'
        for number, label, detail in items
    )
    st.markdown(f'<div class="mh-stepper">{html}</div>', unsafe_allow_html=True)


def render_result(saved, animate=True):
    """Render the prediction card with one-time browser-native animations."""
    prediction = float(saved["prediction"])
    mae = float(saved["mae_test"])
    lower = max(0.0, prediction - mae)
    upper = prediction + mae

    # Add breathing room around the MAE interval so the range meter reads well.
    ruler_min = max(0.0, lower - mae * 0.55)
    ruler_max = upper + mae * 0.55
    ruler_span = max(ruler_max - ruler_min, 1.0)
    lower_pct = max(0.0, min(100.0, ((lower - ruler_min) / ruler_span) * 100))
    upper_pct = max(0.0, min(100.0, ((upper - ruler_min) / ruler_span) * 100))
    prediction_pct = max(0.0, min(100.0, ((prediction - ruler_min) / ruler_span) * 100))
    range_width = max(0.0, upper_pct - lower_pct)

    warning_html = ""

    duration = 2300 if animate else 0
    entrance_duration = 650 if animate else 0
    animation_class = "animate-result" if animate else ""
    should_animate = "true" if animate else "false"

    html = f"""
    <div class="housing-result-root">
      <style>
        * {{ box-sizing:border-box; }}
        body {{ margin:0; background:transparent; font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#1E293B; }}

        .housing-result-card {{
          position:relative; width:100%; padding:28px 30px 26px; overflow:hidden;
          background:linear-gradient(135deg,#FFFFFF 0%,#F8FFFD 60%,#F0FDFA 100%);
          border:1px solid #99F6E4; border-left:5px solid #0D9488; border-radius:24px;
          box-shadow:0 12px 30px rgba(23,35,59,.10);
        }}
        .housing-result-card.animate-result {{
          animation:resultEntrance {entrance_duration}ms cubic-bezier(.22,1,.36,1) both;
        }}
        @keyframes resultEntrance {{ from {{ opacity:0; transform:translateY(18px); }} to {{ opacity:1; transform:translateY(0); }} }}

        .result-top {{ display:flex; justify-content:space-between; align-items:flex-start; gap:28px; }}
        .result-main {{ flex:1; min-width:0; }}
        .result-cap {{ color:#0D9488; font-size:12px; font-weight:850; letter-spacing:.11em; text-transform:uppercase; }}
        .result-price {{ color:#0F172A; font-size:clamp(42px,6vw,62px); line-height:1; font-weight:850; margin:12px 0 10px; letter-spacing:-.035em; font-variant-numeric:tabular-nums; }}
        .result-location {{ color:#475569; font-size:15px; line-height:1.5; }}
        .model-badge {{ min-width:205px; padding:15px 17px; background:rgba(255,255,255,.92); border:1px solid #DDE3EC; border-radius:15px; }}
        .model-kicker {{ color:#667085; font-size:11px; font-weight:750; letter-spacing:.09em; text-transform:uppercase; }}
        .model-name {{ color:#0F172A; font-size:16px; font-weight:800; margin-top:6px; overflow-wrap:anywhere; }}

        .range-panel {{ margin-top:25px; padding:18px 18px 17px; background:#FFFFFF; border:1px solid #DDE3EC; border-radius:16px; }}
        .range-header {{ display:flex; justify-content:space-between; gap:20px; align-items:center; margin-bottom:18px; }}
        .range-title {{ color:#334155; font-size:13px; font-weight:800; }}
        .range-value {{ padding:6px 10px; color:#0F766E; background:#CCFBF1; border-radius:9px; font-size:12px; font-weight:800; white-space:nowrap; }}
        .ruler {{ position:relative; height:72px; margin:0 7px; }}
        .track {{ position:absolute; left:0; right:0; top:29px; height:8px; border-radius:999px; background:#E8EDF3; overflow:hidden; }}
        .track-shine {{ position:absolute; inset:0; width:100%; opacity:.55; background:linear-gradient(90deg,transparent,rgba(255,255,255,.75),transparent); transform:translateX(-100%); }}
        .animate-result .track-shine {{ animation:shine 1900ms ease 520ms 1; }}
        @keyframes shine {{ to {{ transform:translateX(100%); }} }}
        .expected-range {{ position:absolute; top:29px; left:{lower_pct:.4f}%; height:8px; width:{range_width:.4f}%; border-radius:999px; background:linear-gradient(90deg,#99F6E4,#2DD4BF); transform-origin:center; }}
        .animate-result .expected-range {{ animation:rangeExpand {duration}ms cubic-bezier(.22,1,.36,1) both; }}
        @keyframes rangeExpand {{ from {{ transform:scaleX(0); opacity:.25; }} to {{ transform:scaleX(1); opacity:1; }} }}
        .prediction-marker {{ position:absolute; top:18px; left:{prediction_pct:.4f}%; width:22px; height:22px; border-radius:50%; background:#0D9488; border:4px solid #FFFFFF; box-shadow:0 2px 7px rgba(15,118,110,.30); transform:translateX(-50%); }}
        .prediction-marker::after {{ content:""; position:absolute; width:7px; height:7px; border-radius:50%; background:#FFFFFF; left:50%; top:50%; transform:translate(-50%,-50%); }}
        .animate-result .prediction-marker {{ animation:markerSlide {duration}ms cubic-bezier(.22,1,.36,1) both; }}
        @keyframes markerSlide {{ from {{ left:0%; opacity:0; }} to {{ left:{prediction_pct:.4f}%; opacity:1; }} }}
        .marker-line {{ position:absolute; top:39px; left:{prediction_pct:.4f}%; height:10px; width:2px; background:#0D9488; transform:translateX(-50%); }}
        .marker-label {{ position:absolute; top:51px; left:{prediction_pct:.4f}%; transform:translateX(-50%); color:#0F766E; font-size:11px; font-weight:850; white-space:nowrap; }}
        .ruler-min,.ruler-max {{ position:absolute; top:0; color:#7C8799; font-size:11px; font-weight:650; }}
        .ruler-min {{ left:0; }} .ruler-max {{ right:0; }}

        .stats {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-top:18px; }}
        .stat {{ padding:13px 14px; background:rgba(255,255,255,.94); border:1px solid #DDE3EC; border-radius:14px; }}
        .stat-k {{ color:#667085; font-size:10px; font-weight:800; letter-spacing:.07em; text-transform:uppercase; }}
        .stat-v {{ color:#0F172A; margin-top:5px; font-size:15px; font-weight:850; overflow-wrap:anywhere; }}
        .animate-result .stat {{ opacity:0; animation:statEntrance 520ms ease forwards; }}
        .animate-result .stat:nth-child(1) {{ animation-delay:900ms; }}
        .animate-result .stat:nth-child(2) {{ animation-delay:1120ms; }}
        .animate-result .stat:nth-child(3) {{ animation-delay:1340ms; }}
        .animate-result .stat:nth-child(4) {{ animation-delay:1560ms; }}
        @keyframes statEntrance {{ from {{ opacity:0; transform:translateY(8px); }} to {{ opacity:1; transform:translateY(0); }} }}

        .result-warning {{ display:flex; gap:11px; margin-top:14px; padding:13px 15px; color:#92400E; background:#FFFBEB; border:1px solid #FDE68A; border-radius:13px; font-size:13px; line-height:1.5; }}
        .warning-icon {{ display:flex; align-items:center; justify-content:center; flex:0 0 23px; width:23px; height:23px; border-radius:50%; background:#FEF3C7; color:#92400E; font-weight:900; }}
        .result-disclaimer {{ color:#667085; margin-top:14px; font-size:12px; }}

        @media (max-width:720px) {{
          .housing-result-card {{ padding:22px 19px; }}
          .result-top {{ flex-direction:column; }}
          .model-badge {{ width:100%; min-width:0; }}
          .stats {{ grid-template-columns:1fr 1fr; }}
          .result-price {{ font-size:42px; }}
        }}
        @media (max-width:460px) {{
          .stats {{ grid-template-columns:1fr; }}
          .range-header {{ align-items:flex-start; flex-direction:column; gap:7px; }}
        }}
        @media (prefers-reduced-motion:reduce) {{
          *,*::before,*::after {{ animation-duration:.01ms!important; animation-iteration-count:1!important; transition-duration:.01ms!important; }}
        }}
      </style>

      <div class="housing-result-card {animation_class}">
        <div class="result-top">
          <div class="result-main">
            <div class="result-cap">Estimated median price</div>
            <div class="result-price" id="animatedHousePrice">RM {prediction:,.0f}</div>
            <div class="result-location">{escape(saved["area"])}, {escape(saved["state"])} &nbsp;·&nbsp; {escape(saved["ptype"])} &nbsp;·&nbsp; {escape(saved["tenure"])}</div>
          </div>
          <div class="model-badge">
            <div class="model-kicker">Selected model</div>
            <div class="model-name">{escape(saved["model_name"])}</div>
          </div>
        </div>

        <div class="range-panel">
          <div class="range-header">
            <div class="range-title">Expected price range</div>
            <div class="range-value">RM {lower/1000:,.0f}K – RM {upper/1000:,.0f}K</div>
          </div>
          <div class="ruler">
            <div class="ruler-min">RM {ruler_min/1000:,.0f}K</div>
            <div class="ruler-max">RM {ruler_max/1000:,.0f}K</div>
            <div class="track"><div class="track-shine"></div></div>
            <div class="expected-range"></div>
            <div class="prediction-marker"></div>
            <div class="marker-line"></div>
            <div class="marker-label">Prediction</div>
          </div>
        </div>

        <div class="stats">
          <div class="stat"><div class="stat-k">Location</div><div class="stat-v">{escape(saved["state"])}</div></div>
          <div class="stat"><div class="stat-k">Median PSF input</div><div class="stat-v">RM {saved["psf"]:,.0f}</div></div>
          <div class="stat"><div class="stat-k">Test MAE</div><div class="stat-v">RM {saved["mae_test"]/1000:,.1f}K</div></div>
          <div class="stat"><div class="stat-k">Test R²</div><div class="stat-v">{saved["r2_test"]:.3f}</div></div>
        </div>
        {warning_html}
        <div class="result-disclaimer">This is an estimated median price, not a final market valuation.</div>
      </div>

      <script>
        (function() {{
          const shouldAnimate = {should_animate};
          const finalPrice = {prediction:.8f};
          const duration = {duration};
          const priceNode = document.getElementById("animatedHousePrice");
          if (!priceNode) return;

          function formatRM(value) {{
            return "RM " + Math.round(value).toLocaleString("en-MY");
          }}
          if (!shouldAnimate || duration === 0) {{
            priceNode.textContent = formatRM(finalPrice);
            return;
          }}

          priceNode.textContent = "RM 0";
          const startTime = performance.now();
          function easeOutCubic(t) {{ return 1 - Math.pow(1 - t, 3); }}
          function animatePrice(now) {{
            const progress = Math.min((now - startTime) / duration, 1);
            priceNode.textContent = formatRM(finalPrice * easeOutCubic(progress));
            if (progress < 1) requestAnimationFrame(animatePrice);
            else priceNode.textContent = formatRM(finalPrice);
          }}
          requestAnimationFrame(animatePrice);
        }})();
      </script>
    </div>
    """

    component_height = 455
    st.components.v1.html(html, height=component_height, scrolling=False)



def attach_spider_fly_marker(
    malaysia_map,
    fly_request,
    current_state,
    current_area,
    searched_point=None,
):
    """Add a reliable Spider-Man-style navigator to the Folium map.

    The mascot itself is created as a REAL Folium ``Marker``/``DivIcon`` in
    Python, so it is visible even if the optional movement JavaScript fails.
    JavaScript is used only for the smooth flight, speech updates, camera
    ``flyTo`` motion while the idle mascot stays attached to its map coordinate.

    This is intentionally different from the previous implementation, which
    tried to create the marker from injected JavaScript.  In Streamlit/Folium
    that can race the Leaflet map initialisation and leave the mascot missing.
    """
    from branca.element import MacroElement, Template
    import json as _json

    has_flight = bool(fly_request)

    # Keep the mascot attached to a real geographic coordinate across every
    # Streamlit rerun.  An exact searched POI/address has highest priority;
    # otherwise use the selected area's verified coordinate, then the state,
    # then the initial Malaysia map centre.  Property/model input changes must
    # never move Spider-Man away from the selected location.
    default_center = list(getattr(malaysia_map, "location", None) or [4.2105, 108.9758])
    resting_center = list(default_center)
    if searched_point:
        try:
            resting_center = [float(searched_point["lat"]), float(searched_point["lon"])]
        except (KeyError, TypeError, ValueError):
            searched_point = None
    if not searched_point and current_area and current_state:
        area_coords, _ = get_area_map_coords(current_area, current_state)
        if area_coords:
            resting_center = list(area_coords)
    elif not searched_point and current_state and current_state in STATE_COORDS:
        resting_center = list(STATE_COORDS[current_state])

    if has_flight:
        start_center = list(fly_request.get("from_center", default_center))
        target_center = list(fly_request.get("to_center", resting_center))
        target_zoom = int(fly_request.get("to_zoom", 9))
        duration = max(1.2, float(fly_request.get("duration", 3.0)))
    else:
        start_center = list(resting_center)
        target_center = list(resting_center)
        target_zoom = int(getattr(malaysia_map, "options", {}).get("zoom", 9) or 9)
        duration = 0.0

    if current_area:
        destination_label = str(current_area)
        arrived_message = f"I have arrived {current_area}!"
    elif current_state:
        destination_label = str(current_state)
        arrived_message = f"I have arrived {current_state}! Pick an area."
    else:
        destination_label = ""
        arrived_message = "Where do we want to go?"

    initial_message = (
        f"Flying to {destination_label}..."
        if has_flight and destination_label
        else arrived_message
    )

    # ------------------------------------------------------------------
    # REAL FOLIUM MARKER - this exists even without custom JavaScript.
    # ------------------------------------------------------------------
    spider_html = f"""
    <div class="mh-spider-wrap">
      <div class="mh-spider-bubble">{escape(initial_message)}</div>
      <svg class="mh-spider-mask" viewBox="0 0 72 72" aria-hidden="true">
        <defs>
          <linearGradient id="mhSpiderMaskGradient" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stop-color="#FB3657"></stop>
            <stop offset="1" stop-color="#C4143C"></stop>
          </linearGradient>
        </defs>

        <!-- subtle blue outer ring so the icon still reads as Spider-Man on a map -->
        <circle cx="36" cy="36" r="31" fill="#173B7A" opacity=".98"></circle>
        <circle cx="36" cy="36" r="27.5" fill="url(#mhSpiderMaskGradient)"
                stroke="#172554" stroke-width="2.2"></circle>

        <!-- web pattern -->
        <path d="M36 9v54M9 36h54
                 M15 21c13 6 29 6 42 0
                 M15 51c13-6 29-6 42 0
                 M21 13c2.5 13 2.5 33 0 46
                 M51 13c-2.5 13-2.5 33 0 46"
              fill="none" stroke="#172554" stroke-width="1.25" opacity=".68"></path>
        <path d="M15 15L57 57M57 15L15 57"
              fill="none" stroke="#172554" stroke-width="1.05" opacity=".38"></path>

        <!-- eyes -->
        <path d="M19 31c5-8.7 10.7-11.6 16.1-12.1-.8 10-5.3 17-13.4 21.3z"
              fill="#FFFFFF" stroke="#172554" stroke-width="2.4"></path>
        <path d="M53 31c-5-8.7-10.7-11.6-16.1-12.1.8 10 5.3 17 13.4 21.3z"
              fill="#FFFFFF" stroke="#172554" stroke-width="2.4"></path>
      </svg>
    </div>
    """

    spider_icon = folium.DivIcon(
        html=spider_html,
        icon_size=(72, 72),
        icon_anchor=(36, 36),
        class_name="mh-spider-divicon",
    )
    spider_marker = folium.Marker(
        location=start_center,
        icon=spider_icon,
        interactive=False,
        keyboard=False,
        draggable=False,
        z_index_offset=10000,
        rise_on_hover=False,
    ).add_to(malaysia_map)

    # Put styling in the document head.  The real marker above will use it even
    # if the animation macro cannot run for any reason.
    css = r"""
    <style>
      .mh-spider-divicon,
      .mh-spider-divicon * {
        pointer-events:none !important;
        user-select:none !important;
      }
      .leaflet-marker-icon.mh-spider-divicon {
        background:transparent !important;
        border:none !important;
        overflow:visible !important;
      }
      .mh-spider-wrap {
        position:relative;
        width:72px;
        height:72px;
        overflow:visible;
        filter:drop-shadow(0 7px 9px rgba(23,35,59,.24));
        transform-origin:50% 68%;
      }
      .mh-spider-mask {
        display:block;
        width:72px;
        height:72px;
        transform:rotate(180deg);
        transform-origin:50% 50%;
      }
      .mh-spider-bubble {
        position:absolute;
        left:50%;
        bottom:79px;
        transform:translateX(-50%);
        min-width:170px;
        max-width:235px;
        padding:9px 12px;
        color:#172033;
        background:rgba(255,255,255,.98);
        border:1px solid #D7DEE9;
        border-radius:14px;
        box-shadow:0 9px 22px rgba(23,35,59,.14);
        font:800 12px/1.3 Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
        text-align:center;
        white-space:normal;
      }
      .mh-spider-bubble::after {
        content:"";
        position:absolute;
        left:50%;
        bottom:-7px;
        width:12px;
        height:12px;
        transform:translateX(-50%) rotate(45deg);
        background:#FFFFFF;
        border-right:1px solid #D7DEE9;
        border-bottom:1px solid #D7DEE9;
      }
      .mh-spider-wrap.mh-flying {
        animation:mhSpiderSwing .44s ease-in-out infinite alternate;
      }
      .mh-spider-wrap.mh-arrived {
        animation:mhSpiderArrive .68s cubic-bezier(.22,1,.36,1) 1;
      }
      @keyframes mhSpiderSwing {
        from { transform:rotate(-10deg) translateY(2px) scale(.96); }
        to   { transform:rotate(10deg) translateY(-8px) scale(1.04); }
      }
      @keyframes mhSpiderArrive {
        0%   { transform:scale(.78) rotate(-8deg); }
        55%  { transform:scale(1.20) rotate(5deg); }
        100% { transform:scale(1) rotate(0); }
      }
      @media (max-width:650px) {
        .mh-spider-wrap,
        .mh-spider-mask { width:60px; height:60px; }
        .mh-spider-bubble {
          bottom:67px;
          min-width:145px;
          max-width:190px;
          font-size:11px;
        }
      }
      @media (prefers-reduced-motion:reduce) {
        .mh-spider-wrap { animation:none !important; }
      }
    </style>
    """
    malaysia_map.get_root().header.add_child(folium.Element(css))

    # ------------------------------------------------------------------
    # MOTION MACRO - runs AFTER Leaflet, the map, and this real marker exist.
    # ------------------------------------------------------------------
    map_var = malaysia_map.get_name()
    marker_var = spider_marker.get_name()
    destination_js = _json.dumps(destination_label)
    arrived_message_js = _json.dumps(arrived_message)
    has_flight_js = "true" if has_flight else "false"
    target_lat = float(target_center[0])
    target_lon = float(target_center[1])

    js = f"""
    (function() {{
      const mapObj = {map_var};
      const spiderMarker = {marker_var};
      const hasFlight = {has_flight_js};
      const target = L.latLng({target_lat:.7f}, {target_lon:.7f});
      const targetZoom = {target_zoom};
      const durationSec = {duration:.3f};
      const durationMs = Math.max(700, durationSec * 1000);
      const destination = {destination_js};
      const arrivedMessage = {arrived_message_js};

      let flying = false;
      let trail = null;
      let frameId = null;

      function markerElement() {{
        return spiderMarker && spiderMarker.getElement ? spiderMarker.getElement() : null;
      }}
      function wrapElement() {{
        const el = markerElement();
        return el ? el.querySelector('.mh-spider-wrap') : null;
      }}
      function bubbleElement() {{
        const el = markerElement();
        return el ? el.querySelector('.mh-spider-bubble') : null;
      }}
      function setMessage(value) {{
        const bubble = bubbleElement();
        if (bubble) bubble.textContent = value;
      }}
      function easeInOutCubic(t) {{
        return t < 0.5
          ? 4 * t * t * t
          : 1 - Math.pow(-2 * t + 2, 3) / 2;
      }}

      function startFlight() {{
        if (!hasFlight || flying || !spiderMarker) return;

        flying = true;
        const start = spiderMarker.getLatLng();
        const deltaLat = target.lat - start.lat;
        const deltaLng = target.lng - start.lng;
        const distance = Math.sqrt(deltaLat * deltaLat + deltaLng * deltaLng);
        const arc = Math.min(0.65, Math.max(0.035, distance * 0.075));
        const started = performance.now();
        let lastTrailBucket = -1;

        const wrap = wrapElement();
        if (wrap) {{
          wrap.classList.remove('mh-arrived');
          wrap.classList.add('mh-flying');
        }}
        setMessage(destination ? ('Flying to ' + destination + '...') : 'Here we go!');

        trail = L.polyline([start], {{
          color:'#E11D48',
          weight:3,
          opacity:.52,
          dashArray:'6 10',
          lineCap:'round',
          interactive:false
        }}).addTo(mapObj);

        // The camera begins just after the mascot launches.  This macro runs in
        // Folium's script phase, so mapObj is already available and there is no
        // loading race / wait loop.
        window.setTimeout(function() {{
          try {{
            mapObj.stop();
            mapObj.flyTo(target, targetZoom, {{
              animate:true,
              duration:durationSec,
              easeLinearity:0.16,
              noMoveStart:false
            }});
          }} catch (e) {{
            // The real marker and normal map remain usable even if flyTo fails.
          }}
        }}, 180);

        function tick(now) {{
          const raw = Math.min((now - started) / durationMs, 1);
          const eased = easeInOutCubic(raw);
          const lift = Math.sin(Math.PI * raw) * arc;
          const lat = start.lat + deltaLat * eased + lift;
          const lng = start.lng + deltaLng * eased;
          const point = L.latLng(lat, lng);

          spiderMarker.setLatLng(point);

          const bucket = Math.floor(raw * 40);
          if (trail && bucket !== lastTrailBucket) {{
            trail.addLatLng(point);
            lastTrailBucket = bucket;
          }}

          if (raw < 1) {{
            frameId = requestAnimationFrame(tick);
            return;
          }}

          spiderMarker.setLatLng(target);
          flying = false;

          const arrivedWrap = wrapElement();
          if (arrivedWrap) {{
            arrivedWrap.classList.remove('mh-flying');
            arrivedWrap.classList.add('mh-arrived');
          }}
          setMessage(arrivedMessage);

          window.setTimeout(function() {{
            if (trail) {{
              try {{ mapObj.removeLayer(trail); }} catch (e) {{}}
              trail = null;
            }}
          }}, 1200);
        }}

        frameId = requestAnimationFrame(tick);
      }}

      if (hasFlight) {{
        // Marker already exists; this delay is just for the user to see the
        // starting position before Spider-Man launches.
        window.setTimeout(startFlight, 650);
      }} else {{
        // Static / idle state: keep him anchored to the selected Lat/Lon.
        spiderMarker.setLatLng(target);
        setMessage(arrivedMessage);
      }}
    }})();
    """

    class _SpiderMotion(MacroElement):
        def __init__(self, javascript: str):
            super().__init__()
            self._name = "SpiderMotion"
            self.javascript = javascript
            self._template = Template(
                "{% macro script(this, kwargs) %}"
                "{{ this.javascript | safe }}"
                "{% endmacro %}"
            )

    _SpiderMotion(js).add_to(malaysia_map)

def prediction_page(data, results):
    recommended = selected_model_name(results)
    default_model = model_default_name(results)
    model_options = results["Model"].astype(str).tolist()
    available_states = sorted(STATE_COORDS)
    ptypes = sorted(data["Primary_Type"].dropna().astype(str).unique())
    tenure_options = sorted(data["Tenure"].dropna().astype(str).unique())
    defaults = {
        "selected_state": None,
        "selected_area": None,
        "selected_ptype": ptypes[0],
        "selected_tenure": tenure_options[0],
        "address_input": "",
        "address_feedback": None,
        "selected_prediction_model": default_model,
        "last_prediction": None,
        "last_map_popup": None,
        "searched_point": None,
        "map_fly_request": None,
        "location_candidates": [],
        "location_candidate_query": "",
        "location_candidate_choice": None,
        "saved_scenarios": [],
        "just_predicted": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    if st.session_state.get("selected_prediction_model") not in model_options:
        st.session_state["selected_prediction_model"] = default_model

    # A map event arrives near the end of a run. Apply it once at the beginning
    # of the next run so a click cannot create a repeated rerun loop.
    pending_location = st.session_state.pop("pending_location", None)
    if pending_location:
        pending_state, pending_area = pending_location
        st.session_state["selected_state"] = pending_state
        st.session_state["selected_area"] = pending_area
        st.session_state["searched_point"] = None

    current_state = st.session_state["selected_state"]
    current_area = st.session_state["selected_area"]

    # ---------------- MODEL SELECTION MOVED TO TOP ----------------
    field_label("Prediction model")
    selected_model = st.selectbox(
        "Prediction model",
        model_options,
        index=model_options.index(st.session_state["selected_prediction_model"]),
        key="selected_prediction_model",
        label_visibility="collapsed",
        on_change=clear_last_prediction,
        help="Random Forest is the default model. You can still compare the other saved models.",
    )
    selected_metrics = results.loc[results["Model"].eq(selected_model)].iloc[0]
    model_note = f"Default model: {default_model}"
    if selected_model == recommended:
        model_note += f" · Recommended by evaluation · Test MAE RM {selected_metrics['MAE_test']/1000:,.1f}K · Test R² {selected_metrics['R2_test']:.3f}"
    else:
        model_note += f" · Selected Test MAE RM {selected_metrics['MAE_test']/1000:,.1f}K · Test R² {selected_metrics['R2_test']:.3f} · Evaluation recommended: {recommended}"
    st.caption(model_note)

    # ---------------- LOCATION SELECTION: OLD UI DESIGN, FASTER LOGIC ----------------
    st.markdown(
        "<div class='mh-section-head'><div class='mh-step'>1</div><div>"
        "<h3 class='mh-section-title'>Location Selection</h3>"
        "<p class='mh-section-note'>Choose a state, then select an area from the nationwide Malaysia coverage.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    search_col, button_col = st.columns([5, 1.25])
    with search_col:
        st.text_input(
            "Enter your address or postcode to auto-detect location, or click the map below:",
            placeholder="e.g. 45400, Setapak, or a building/condo name",
            key="address_input",
        )
    with button_col:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.button(
            "Search Location",
            type="primary",
            use_container_width=True,
            on_click=analyze_address,
            args=(available_states,),
            key="search_location",
        )

    feedback = st.session_state.get("address_feedback")
    if feedback:
        kind, msg = feedback
        if kind == "success":
            st.success(msg, icon="✅")
        elif kind == "info":
            st.info(msg, icon="ℹ️")
        else:
            st.warning(msg, icon="⚠️")

    # Ambiguous POI/building searches are never auto-accepted. The candidate
    # list lives in session_state, so changing the selectbox does not rerun the
    # search or lose the selected option.
    candidates = st.session_state.get("location_candidates", [])
    if candidates:
        st.markdown("**Choose the correct location:**")
        candidate_by_id = {}
        candidate_ids = []
        for idx, candidate in enumerate(candidates):
            candidate_id = candidate.get("candidate_id") or f"candidate_{idx}"
            candidate_by_id[candidate_id] = candidate
            candidate_ids.append(candidate_id)

        current_choice = st.session_state.get("location_candidate_choice")
        if current_choice not in candidate_by_id:
            st.session_state["location_candidate_choice"] = candidate_ids[0]

        chosen_id = st.selectbox(
            "Possible locations",
            candidate_ids,
            format_func=lambda candidate_id: candidate_label(candidate_by_id[candidate_id]),
            key="location_candidate_choice",
            label_visibility="collapsed",
        )

        choose_col, cancel_col = st.columns([1.35, 1])
        with choose_col:
            if st.button(
                "Use selected location",
                type="primary",
                use_container_width=True,
                key="use_location_candidate",
            ):
                chosen_candidate = candidate_by_id.get(chosen_id)
                if chosen_candidate and apply_geocode_candidate(chosen_candidate, available_states):
                    st.rerun()
        with cancel_col:
            if st.button("Cancel search", use_container_width=True, key="cancel_location_candidates"):
                st.session_state["location_candidates"] = []
                st.session_state["location_candidate_query"] = ""
                st.session_state.pop("location_candidate_choice", None)
                st.session_state["address_feedback"] = None
                st.rerun()

    if HAS_INTERACTIVE_MAP:
        # A successful search/click can request a one-time Leaflet fly animation.
        # The request is popped here so ordinary Streamlit reruns remain static.
        fly_request = st.session_state.pop("map_fly_request", None)

        final_map_center = st.session_state.get(
            "map_center",
            [4.2105, 108.9758] if not current_state else STATE_COORDS.get(current_state, [4.2105, 108.9758]),
        )
        final_map_zoom = st.session_state.get("map_zoom", 6 if not current_state else 9)

        if fly_request:
            map_center = fly_request.get("from_center", final_map_center)
            map_zoom = fly_request.get("from_zoom", max(6, int(final_map_zoom) - 3))
        else:
            map_center = final_map_center
            map_zoom = final_map_zoom

        malaysia_map = folium.Map(
            location=map_center,
            zoom_start=map_zoom,
            tiles="OpenStreetMap",
            control_scale=True,
            zoom_control=True,
            prefer_canvas=True,
            zoom_animation=True,
            fade_animation=True,
            marker_zoom_animation=True,
        )

        if not current_state:
            for state_name in available_states:
                folium.CircleMarker(
                    location=STATE_COORDS[state_name],
                    radius=11,
                    color="#93C5FD",
                    weight=2,
                    fill=True,
                    fill_color="#2F6FED",
                    fill_opacity=0.82,
                    tooltip=folium.Tooltip(f"<b>{state_name}</b> (Click to select state)", sticky=True),
                    popup=f"STATE:{state_name}",
                ).add_to(malaysia_map)
        else:
            marker_cluster = MarkerCluster(name="Areas", options={"chunkedLoading": True, "disableClusteringAtZoom": 12}).add_to(malaysia_map)
            marker_rows = list(get_area_marker_data(current_state))
            if current_area and current_area not in {row[0] for row in marker_rows}:
                coords, approx = get_area_map_coords(current_area, current_state)
                if coords:
                    marker_rows.append((current_area, coords[0], coords[1], approx, get_area_source(current_state, current_area)))

            searched_point = st.session_state.get("searched_point")

            for disp_area, lat, lon, is_approx, area_kind in marker_rows:
                # When a building/POI search gives an exact coordinate, keep the
                # generic housing-area pin as a normal reference pin.  The exact
                # searched coordinate becomes the selected point instead.  This
                # avoids showing two different green "selected" locations.
                is_sel = (
                    disp_area == current_area
                    and not searched_point
                )
                pin_note = "Approximate position · click to select" if is_approx else "Click to select"
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=12 if is_sel else 8,
                    color="#18875D" if is_sel else ("#98A2B3" if is_approx else "#C47A10"),
                    weight=3 if is_sel else 2,
                    fill=True,
                    fill_color="#10B981" if is_sel else ("#D0D5DD" if is_approx else "#F59E0B"),
                    fill_opacity=0.9 if is_sel else 0.75,
                    tooltip=folium.Tooltip(f"<b>{disp_area}</b><br>{area_kind} · {pin_note}", sticky=True),
                    popup=f"AREA:{disp_area}",
                ).add_to(marker_cluster)

            if searched_point and current_state:
                folium.CircleMarker(
                    location=[searched_point["lat"], searched_point["lon"]],
                    radius=8,
                    color="#0F766E",
                    weight=3,
                    fill=True,
                    fill_color="#10B981",
                    fill_opacity=0.95,
                    tooltip=folium.Tooltip(
                        f"<b>Selected searched location</b><br>{escape(str(searched_point.get('label', '')))}"
                        + (f"<br>Housing area: {escape(str(current_area))}, {escape(str(current_state))}" if current_area else ""),
                        sticky=True,
                    ),
                    popup="SEARCHED_POINT",
                ).add_to(malaysia_map)

        # Spider-Man-style navigator lives INSIDE the Leaflet map. It is
        # non-interactive, so all original state/area pins remain clickable.
        # It also owns the smooth map fly animation; no loading overlay is used.
        attach_spider_fly_marker(
            malaysia_map,
            fly_request,
            current_state,
            current_area,
            st.session_state.get("searched_point"),
        )

        st.markdown("<div class='mh-map-wrap'>", unsafe_allow_html=True)
        map_kwargs = dict(height=470, use_container_width=True, key="malaysia_map")
        if "returned_objects" in inspect.signature(st_folium).parameters:
            map_kwargs["returned_objects"] = ["last_object_clicked_popup"]
        map_event = st_folium(malaysia_map, **map_kwargs)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='mh-map-legend'>"
            "<span class='legend-title'>Map pins:</span>"
            "<span class='legend-item'><span class='legend-dot state-dot'></span>State</span>"
            "<span class='legend-item'><span class='legend-dot verified-dot'></span>Known map location</span>"
            "<span class='legend-item'><span class='legend-dot approx-dot'></span>Estimated map location</span>"
            "<span class='legend-item'><span class='legend-dot selected-dot'></span>Selected area</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        popup_txt = (map_event or {}).get("last_object_clicked_popup")
        if popup_txt and popup_txt != st.session_state.get("last_map_popup"):
            st.session_state["last_map_popup"] = popup_txt
            if popup_txt.startswith("STATE:"):
                clicked_st = popup_txt.split(":", 1)[1]
                if clicked_st != current_state:
                    old_center = list(st.session_state.get("map_center", [4.2105, 108.9758]))
                    old_zoom = int(st.session_state.get("map_zoom", 6))
                    target = list(STATE_COORDS.get(clicked_st, [4.2105, 108.9758]))
                    st.session_state["pending_location"] = (clicked_st, None)
                    st.session_state["map_center"] = target
                    st.session_state["map_zoom"] = 9
                    st.session_state["map_fly_request"] = {
                        "from_center": old_center, "from_zoom": old_zoom,
                        "to_center": target, "to_zoom": 9, "duration": 3.0,
                    }
                    st.session_state["searched_point"] = None
                    clear_last_prediction()
                    st.rerun()
            elif popup_txt.startswith("AREA:"):
                clicked_area = popup_txt.split(":", 1)[1]
                if clicked_area != current_area:
                    old_center = list(st.session_state.get("map_center", STATE_COORDS.get(current_state, [4.2105, 108.9758])))
                    old_zoom = int(st.session_state.get("map_zoom", 9))
                    st.session_state["pending_location"] = (current_state, clicked_area)
                    coords, _ = get_area_map_coords(clicked_area, current_state)
                    target = list(coords) if coords else old_center
                    st.session_state["map_center"] = target
                    st.session_state["map_zoom"] = 12
                    st.session_state["map_fly_request"] = {
                        "from_center": old_center, "from_zoom": old_zoom,
                        "to_center": target, "to_zoom": 12, "duration": 2.8,
                    }
                    st.session_state["searched_point"] = None
                    clear_last_prediction()
                    st.rerun()
    else:
        st.error("Interactive map packages are unavailable. Install folium and streamlit-folium to select a location.")

    current_state = st.session_state["selected_state"]
    current_area = st.session_state["selected_area"]
    col_loc1, col_loc2 = st.columns([5, 1.25])
    with col_loc1:
        st.markdown(
            f"<div class='mh-chiprow'><span class='mh-chip'>📍 <strong>State:</strong> {escape(current_state or 'Not Selected')}</span>"
            f"<span class='mh-chip'>🗺️ <strong>Area:</strong> {escape(current_area or 'Not Selected')}</span></div>",
            unsafe_allow_html=True,
        )
    with col_loc2:
        st.button("Reset Location", use_container_width=True, on_click=reset_location_state, key="reset_location")
    st.markdown(
        "<div class='mh-help'>ⓘ Map-pin colours describe coordinate accuracy, not dataset membership. "
        "Known map locations use verified coordinates; estimated map locations use a fallback position. "
        "All selectable locations can still be used for nationwide Malaysia prediction.</div>",
        unsafe_allow_html=True,
    )

    # ---------------- PROPERTY DETAILS: OLD UI DESIGN ----------------
    st.markdown('<hr class="mh-rule">', unsafe_allow_html=True)
    st.markdown(
        "<div class='mh-section-head'><div class='mh-step'>2</div><div>"
        "<h3 class='mh-section-title'>Property Details</h3>"
        "<p class='mh-section-note'>Choose the property type and tenure, then enter the independently known median price per square foot.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    field_label("Select Property Type")
    ptype_cols = st.columns(len(ptypes), gap="small")
    for i, pt in enumerate(ptypes):
        with ptype_cols[i]:
            is_sel = pt == st.session_state["selected_ptype"]
            if st.button(
                get_property_label(pt),
                key=f"btn_{pt}",
                type="primary" if is_sel else "secondary",
                use_container_width=True,
            ):
                st.session_state["selected_ptype"] = pt
                clear_last_prediction()
                st.rerun()

    reference = market_reference(
        data,
        current_state,
        current_area,
        st.session_state["selected_ptype"],
        st.session_state["selected_tenure"],
    )

    col_in1, col_in2 = st.columns(2)
    with col_in1:
        field_label("Tenure")
        st.markdown("<div class='tenure-hint'>Ownership status</div>", unsafe_allow_html=True)
        tenure_group, _ = st.columns([3.2, 1.15])
        with tenure_group:
            tenure_cols = st.columns(len(tenure_options), gap="small")
            for idx, tenure_option in enumerate(tenure_options):
                with tenure_cols[idx]:
                    is_tenure_selected = tenure_option == st.session_state["selected_tenure"]
                    if st.button(
                        tenure_option,
                        key=f"tenure_btn_{tenure_option}",
                        type="primary" if is_tenure_selected else "secondary",
                        use_container_width=True,
                    ):
                        st.session_state["selected_tenure"] = tenure_option
                        clear_last_prediction()
                        st.rerun()
        tenure = st.session_state["selected_tenure"]
    with col_in2:
        field_label("Median price per sq ft (RM)")
        psf = st.number_input(
            "PSF",
            min_value=1,
            step=10,
            value=int(round(data["Median_PSF"].median())),
            key="pred_psf",
            label_visibility="collapsed",
            on_change=clear_last_prediction,
        )
        st.caption("Use market median PSF for the selected area and property type.")

    psf_confirmed = st.checkbox(
        "I confirm that Median PSF is independently known before prediction.",
        key="pred_psf_confirmed",
        help="Do not derive PSF from the Median Price being predicted.",
        on_change=clear_last_prediction,
    )

    # The fitted pipelines still require Transactions. It is intentionally not
    # a user input: the app supplies the median from the closest matching
    # historical market reference to keep the interface simple and consistent.
    model_transactions = reference["transactions"]
    area_pool = data
    if current_state:
        area_pool = area_pool[area_pool["State"].eq(current_state)]
    if current_area and len(area_pool):
        matched_pool = area_pool[area_pool["Area_Clean"].eq(clean_area_name(current_area))]
        if len(matched_pool):
            area_pool = matched_pool
    market_median = float(area_pool["Median_PSF"].median()) if len(area_pool) else float(data["Median_PSF"].median())
    observed_min, observed_max = float(data["Median_PSF"].min()), float(data["Median_PSF"].max())
    if psf < observed_min or psf > observed_max:
        st.warning(f"RM {psf:,.0f}/sq ft is outside the observed dataset range of RM {observed_min:,.0f}–RM {observed_max:,.0f}.")
    elif psf < market_median * .5 or psf > market_median * 1.5:
        st.warning(f"This PSF is far from the relevant market median of approximately RM {market_median:,.0f}/sq ft. Please verify it.")

    st.markdown('<br>', unsafe_allow_html=True)
    predict_clicked = st.button("Generate Price Estimate  →", type="primary", use_container_width=True, key="generate_estimate")

    if predict_clicked:
        if not current_state or not current_area:
            st.error("Please select both a State and an Area from the map or address input before predicting.")
        elif not psf_confirmed:
            st.error("Please confirm that Median PSF is independently known before prediction.")
        else:
            try:
                model = load_model(selected_model)
            except Exception as error:
                st.error(
                    f"Unable to load {selected_model}. The saved model file is incompatible or damaged. "
                    "Use the matching requirements.txt and replace that model file before trying again."
                )
                st.caption(f"Technical detail: {type(error).__name__}: {error}")
                return
            area_key = create_area_key(current_state, current_area)
            features = pd.DataFrame([{
                "State": current_state,
                "Area_Key": area_key,
                "Tenure": tenure,
                "Primary_Type": st.session_state["selected_ptype"],
                "Median_PSF": psf,
                "Transactions": model_transactions,
            }])[MODEL_FEATURES]
            with st.spinner("Calculating estimate..."):
                prediction = float(model.predict(features)[0])
            metrics = results.loc[results["Model"].eq(selected_model)].iloc[0]
            st.session_state["last_prediction"] = {
                "prediction": prediction,
                "state": current_state,
                "area": current_area,
                "ptype": st.session_state["selected_ptype"],
                "tenure": tenure,
                "psf": psf,
                "transactions": model_transactions,
                "model_name": selected_model,
                "rmse_test": float(metrics["RMSE_test"]),
                "mae_test": float(metrics["MAE_test"]),
                "r2_test": float(metrics["R2_test"]),
                "dataset_supported": area_key in set(data["Area_Key"].astype(str)),
                "model_supported": model_has_seen_area(model, area_key),
            }
            # Consumed once by the result renderer below. This prevents the
            # count-up/range animation from replaying on unrelated Streamlit reruns.
            st.session_state["just_predicted"] = True

    saved = st.session_state.get("last_prediction")
    if saved:
        just_predicted = st.session_state.pop("just_predicted", False)
        render_result(saved, animate=just_predicted)
        with st.expander("How this estimate is calculated"):
            st.markdown(
                "The selected trained model uses the mapped location, tenure, property type, and Median PSF. "
                "Its required historical transaction statistic is supplied automatically from the closest "
                "matching dataset segment, so the user does not need to enter it. The expected range adds "
                "and subtracts that model's held-out test MAE from the point estimate."
            )
        save_col, clear_col = st.columns([1, 1])
        with save_col:
            if st.button("Save scenario", use_container_width=True, key="save_scenario"):
                scenarios = list(st.session_state.get("saved_scenarios", []))
                signature = (saved["state"], saved["area"], saved["ptype"], saved["tenure"], saved["psf"], saved["model_name"])
                if signature not in {(s["state"], s["area"], s["ptype"], s["tenure"], s["psf"], s["model_name"]) for s in scenarios}:
                    scenarios.append(dict(saved))
                    st.session_state["saved_scenarios"] = scenarios[-3:]
                    st.rerun()
                else:
                    st.info("This scenario is already saved.")
        with clear_col:
            if st.button("Clear saved scenarios", use_container_width=True, key="clear_scenarios"):
                st.session_state["saved_scenarios"] = []
                st.rerun()
    else:
        st.markdown(
            f'<div class="mh-empty"><div class="icon">{svg_icon("money",24)}</div>'
            '<strong>Your estimate will appear here.</strong><br>Select location and property details above, then generate the estimate.</div>',
            unsafe_allow_html=True,
        )

    scenarios = st.session_state.get("saved_scenarios", [])
    if scenarios:
        st.markdown("### Compare saved scenarios")
        comparison = pd.DataFrame([{
            "Location": f"{s['area']}, {s['state']}",
            "Property type": s["ptype"],
            "Tenure": s["tenure"],
            "Median PSF (RM)": s["psf"],
            "Model": s["model_name"],
            "Estimated price (RM)": round(s["prediction"]),
        } for s in scenarios])
        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
            column_config={"Estimated price (RM)": st.column_config.NumberColumn(format="RM %,.0f")},
        )

# ---------------------------------------------------------------------------
# PAGE 2 - MARKET INSIGHTS
# ---------------------------------------------------------------------------
LIVE_INSIGHT_GROUPS = {
    "Data quality": [
        (1, "Raw Median Price distribution", "House prices are strongly right-skewed; the log view makes the same 2,000 records easier to read."),
        (2, "Raw numerical distributions", "Price, PSF and Transactions contain long right tails and potential IQR outliers."),
        (3, "Raw categorical labels", "Tenure and Type contain inconsistent raw labels that need standardisation before modelling."),
        (7, "Extreme values flagged and retained", "Extreme price and PSF records are flagged but kept so the model still covers the full market range."),
        (8, "Retained price distribution", "The original and log views contain the same records; only the display scale changes."),
    ],
    "Data preparation and coverage": [
        (4, "Tenure before and after standardisation", "Equivalent mixed-tenure labels are combined into one consistent Mixed category."),
        (5, "Property Type before and after processing", "Raw Type combinations are converted into a consistent Primary_Type for modelling."),
        (6, "Area labels before and after standardisation", "State-qualified Area_Key keeps same-named locations in different states separate."),
        (9, "Area record-frequency bands", "Many Areas have limited observations, so rare and unseen locations require cautious interpretation."),
    ],
    "Market composition": [
        (10, "Record count by State", "Selangor and Johor contain the most records, while several states have much smaller samples."),
        (11, "Record count by Property Type", "Terrace House has the strongest representation; niche property types have less training evidence."),
        (12, "Landed versus High-Rise composition", "The dataset contains more Landed than High-Rise records; this describes this dataset only."),
    ],
    "Price patterns": [
        (13, "Price distribution by State", "Price levels differ by state, but large within-state spreads show that State alone is not enough."),
        (14, "Price distribution by Area", "Area reveals finer location differences and supports using Area_Key as a model input."),
        (15, "Price by Property Type", "Property types occupy different price bands, although their distributions still overlap."),
        (16, "Price distribution by Tenure", "Leasehold tends to have a lower median price, while the small Mixed group should be read carefully."),
        (17, "Price by broad property category", "Landed properties generally have higher prices, but Category alone cannot explain all variation."),
        (18, "Median PSF distribution by State", "Median PSF differs clearly across states and provides useful location-related market information."),
    ],
    "Relationships and combined effects": [
        (19, "Median PSF against Median Price", "Median PSF has a strong positive relationship with price for both Landed and High-Rise properties."),
        (20, "Transactions against Median Price", "Transactions has only a very weak relationship with price compared with PSF and location."),
        (21, "Correlation matrix", "Median_PSF has the strongest numeric relationship with price, while Transactions contributes very little."),
        (22, "Final feature association with price", "Median PSF is strongest, followed by Area and Property Type; Transactions is weakest."),
        (23, "Median price by State and Category", "The Landed versus High-Rise price difference changes across states, showing a combined effect."),
        (24, "Median price by Property Type and Tenure", "Price varies across Property Type and Tenure combinations; thin groups are masked."),
    ],
}

LIVE_INSIGHT_TAKEAWAYS = {
    "Data quality": "The market contains a long premium tail, so the model must be evaluated on the full price range instead of only the middle of the market.",
    "Data preparation and coverage": "Cleaning preserves all valid records while making Tenure, Type and Area labels consistent for modelling.",
    "Market composition": "The dataset is not evenly distributed across states and property types, so small groups should be interpreted carefully.",
    "Price patterns": "Location and property characteristics clearly separate price levels, but no single categorical feature explains price by itself.",
    "Relationships and combined effects": "Median PSF is the strongest relationship, while Area and Property Type add important structural information.",
}


@st.cache_data(show_spinner=False)
def load_raw_insight_data():
    raw_path = APP_DIR / "malaysia_house_price_data_2025.csv"
    if raw_path.exists():
        return pd.read_csv(raw_path)

    fallback = load_data().copy()
    if "Area_Raw" in fallback.columns and "Area" not in fallback.columns:
        fallback["Area"] = fallback["Area_Raw"]
    return fallback


def render_plotly(fig):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=15, r=15, t=70, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
        hoverlabel=dict(bgcolor="#FFFFFF"),
        legend=dict(title=None),
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "scrollZoom": True,
            "responsive": True,
        },
    )


def _live_iqr_flags(series, k=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    low = q1 - k * iqr
    high = q3 + k * iqr
    return (series < low) | (series > high), low, high


def _live_log_iqr_fences(series, k=1.5):
    import numpy as np

    log_values = np.log1p(series)
    q1 = log_values.quantile(0.25)
    q3 = log_values.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def _live_correlation_ratio(categories, measurements):
    import math
    import numpy as np

    cat = pd.Categorical(categories)
    values = np.asarray(measurements, dtype=float)
    grand_mean = values.mean()
    numerator = 0.0
    for level in range(len(cat.categories)):
        group = values[cat.codes == level]
        if len(group):
            numerator += len(group) * (group.mean() - grand_mean) ** 2
    denominator = ((values - grand_mean) ** 2).sum()
    return math.sqrt(numerator / denominator) if denominator else 0.0


def _upper_whisker(series):
    values = pd.Series(series).dropna().astype(float)
    if values.empty:
        return 0.0
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    inside = values[values <= upper_fence]
    return float(inside.max() if len(inside) else values.max())


def _lower_whisker(series):
    values = pd.Series(series).dropna().astype(float)
    if values.empty:
        return 0.0
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    inside = values[values >= lower_fence]
    return float(inside.min() if len(inside) else values.min())


def _apply_box_axis_range(fig, groups, value_col, pad=1.05):
    maxima = []
    for _, group in groups:
        if len(group):
            maxima.append(_upper_whisker(group[value_col]))
    if maxima:
        fig.update_xaxes(range=[0, max(maxima) * pad])
    return fig


def _histogram_trace(values, bins, name, color, opacity=0.85):
    import numpy as np
    counts, edges = np.histogram(values, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    widths = edges[1:] - edges[:-1]
    return go.Bar(
        x=centres,
        y=counts,
        width=widths,
        name=name,
        marker_color=color,
        opacity=opacity,
        hovertemplate='Range: %{customdata[0]:,.0f} – %{customdata[1]:,.0f}<br>Records: %{y:,}<extra></extra>',
        customdata=np.column_stack([edges[:-1], edges[1:]]),
        showlegend=False,
    )


def build_live_notebook_figure(figure_number, data, raw):
    if not HAS_PLOTLY:
        return None

    import numpy as np
    from plotly.subplots import make_subplots

    # Figure 1 -------------------------------------------------------------
    if figure_number == 1:
        price = raw['Median_Price'].dropna().astype(float)
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"Linear scale (skewness = {price.skew():.2f})",
                f"Log scale (log skewness = {np.log(price).skew():.2f})",
            ),
        )
        linear_edges = np.histogram_bin_edges(price / 1000, bins=60)
        log_edges = np.logspace(np.log10(price.min()), np.log10(price.max()), 60)
        fig.add_trace(_histogram_trace(price / 1000, linear_edges, 'Price', '#2F6FED'), row=1, col=1)
        fig.add_trace(_histogram_trace(price, log_edges, 'Price', '#7C3AED'), row=1, col=2)
        fig.update_xaxes(title_text="Median price (RM '000)", row=1, col=1)
        fig.update_xaxes(title_text='Median price (RM, log scale)', type='log', row=1, col=2)
        fig.update_yaxes(title_text='Records', row=1, col=1)
        fig.update_layout(title='Figure 1 — Raw Median Price distribution on linear and logarithmic scales', bargap=0.02)
        return fig

    # Figure 2 -------------------------------------------------------------
    if figure_number == 2:
        cols = ['Median_Price', 'Median_PSF', 'Transactions']
        fig = make_subplots(rows=3, cols=1, vertical_spacing=0.15)
        for row, col in enumerate(cols, start=1):
            values = raw[col].dropna().astype(float)
            flags, _, _ = _live_iqr_flags(values)
            fig.add_trace(
                go.Box(
                    x=values,
                    orientation='h',
                    boxpoints='outliers',
                    quartilemethod='linear',
                    marker=dict(size=4, color='#667085', opacity=0.30),
                    fillcolor='#FFFFFF',
                    line=dict(color='#667085'),
                    showlegend=False,
                    hovertemplate=f'{col}: %{{x:,.0f}}<extra></extra>',
                    name=col,
                ),
                row=row,
                col=1,
            )
            fig.update_xaxes(type='log', title_text=f'{col} (log axis)', row=row, col=1)
            fig.update_yaxes(showticklabels=False, row=row, col=1)
            fig.add_annotation(
                x=0,
                y=1.08,
                xref=f'x{row if row > 1 else ""} domain',
                yref=f'y{row if row > 1 else ""} domain',
                text=f'{col}: {int(flags.sum())} raw-scale IQR flags',
                showarrow=False,
                xanchor='left',
                font=dict(size=12),
            )
        fig.update_layout(title='Figure 2 — Raw numerical distributions and potential IQR outliers', height=700)
        return fig

    # Figure 3 -------------------------------------------------------------
    if figure_number == 3:
        tenure_counts = raw['Tenure'].value_counts().sort_values()
        type_counts = raw['Type'].value_counts().head(10).sort_values()
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"Tenure: {raw['Tenure'].nunique()} raw labels",
                f"Top 10 of {raw['Type'].nunique()} raw Type strings",
            ),
            horizontal_spacing=0.18,
        )
        fig.add_trace(go.Bar(x=tenure_counts.values, y=tenure_counts.index, orientation='h', marker_color='#2F6FED', text=tenure_counts.values, textposition='outside', hovertemplate='%{y}<br>Records: %{x:,}<extra></extra>', showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=type_counts.values, y=type_counts.index, orientation='h', marker_color='#C47A10', text=type_counts.values, textposition='outside', hovertemplate='%{y}<br>Records: %{x:,}<extra></extra>', showlegend=False), row=1, col=2)
        fig.update_xaxes(title_text='Records', row=1, col=1)
        fig.update_xaxes(title_text='Records', row=1, col=2)
        fig.update_layout(title='Figure 3 — Raw categorical labels', height=520)
        return fig

    # Figure 4 -------------------------------------------------------------
    if figure_number == 4:
        before = raw['Tenure'].value_counts()
        after = data['Tenure'].value_counts()
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Before', 'After'))
        fig.add_trace(go.Bar(x=before.index.astype(str), y=before.values, marker_color='#C47A10', text=before.values, textposition='outside', showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=after.index.astype(str), y=after.values, marker_color='#18875D', text=after.values, textposition='outside', showlegend=False), row=1, col=2)
        fig.update_yaxes(title_text='Rows', row=1, col=1)
        fig.update_xaxes(tickangle=20, row=1, col=1)
        fig.update_xaxes(tickangle=20, row=1, col=2)
        fig.update_layout(title='Figure 4 — Tenure categories before and after standardisation', height=480)
        return fig

    # Figure 5 -------------------------------------------------------------
    if figure_number == 5:
        raw_type_counts = data['Type'].value_counts().head(10).sort_values()
        primary_counts = data['Primary_Type'].value_counts().sort_values()
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Top 10 raw Type strings', 'Deterministic Primary_Type'), horizontal_spacing=0.20)
        fig.add_trace(go.Bar(x=raw_type_counts.values, y=raw_type_counts.index, orientation='h', marker_color='#C47A10', text=raw_type_counts.values, textposition='outside', showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=primary_counts.values, y=primary_counts.index, orientation='h', marker_color='#2F6FED', text=primary_counts.values, textposition='outside', showlegend=False), row=1, col=2)
        fig.update_xaxes(title_text='Rows', range=[0, raw_type_counts.max() * 1.14], row=1, col=1)
        fig.update_xaxes(title_text='Rows', range=[0, primary_counts.max() * 1.14], row=1, col=2)
        fig.update_layout(title='Figure 5 — Property Type before and after processing', height=560)
        return fig

    # Figure 6 -------------------------------------------------------------
    if figure_number == 6:
        summary = pd.DataFrame({
            'Stage': ['Raw Area text', 'Cleaned Area text', 'State-qualified Area_Key'],
            'Distinct labels': [data['Area_Raw'].nunique(), data['Area_Clean'].nunique(), data['Area_Key'].nunique()],
        })
        fig = px.bar(summary, x='Stage', y='Distinct labels', text='Distinct labels', title='Figure 6 — Distinct Area labels before and after standardisation')
        fig.update_traces(marker_color=['#667085', '#2F6FED', '#18875D'], textposition='outside', hovertemplate='%{x}<br>Distinct values: %{y:,}<extra></extra>')
        fig.update_yaxes(title_text='Distinct values', range=[0, summary['Distinct labels'].max() * 1.15])
        fig.update_xaxes(title_text='')
        return fig

    # Figure 7 -------------------------------------------------------------
    if figure_number == 7:
        specs = [
            ('Median_Price', [50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000], ['50K', '100K', '200K', '500K', '1M', '2M', '5M', '10M'], 'Median Price (RM, original-scale labels; log1p used for the IQR rule)'),
            ('Median_PSF', [50, 100, 200, 500, 1000, 2000, 3000], ['50', '100', '200', '500', '1000', '2000', '3000'], 'Median PSF (RM, original-scale labels; log1p used for the IQR rule)'),
        ]
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.20)
        unique_flags = pd.Series(False, index=data.index)
        for row, (col, ticks, tick_labels, xlabel) in enumerate(specs, start=1):
            values = data[col].astype(float)
            values_log = np.log1p(values)
            lower, upper = _live_log_iqr_fences(values)
            flags = (values_log < lower) | (values_log > upper)
            unique_flags = unique_flags | flags
            fig.add_trace(go.Box(x=values_log, orientation='h', boxpoints=False, quartilemethod='linear', fillcolor='#FFFFFF', line=dict(color='#667085'), showlegend=False, name=col), row=row, col=1)
            if flags.any():
                fig.add_trace(go.Scatter(x=values_log[flags], y=[0] * int(flags.sum()), mode='markers', marker=dict(size=6, color='#C63C4A', opacity=0.45), customdata=values[flags], hovertemplate=f'{col}: %{{customdata:,.0f}}<extra></extra>', name=f'Flagged as extreme (retained): {int(flags.sum())}', showlegend=True), row=row, col=1)
            fig.add_vline(x=lower, line_dash='dash', line_width=1.4, line_color='#2F6FED', row=row, col=1)
            fig.add_vline(x=upper, line_dash='dash', line_width=1.4, line_color='#2F6FED', row=row, col=1)
            fig.update_xaxes(tickvals=np.log1p(ticks), ticktext=tick_labels, title_text=xlabel, row=row, col=1)
            fig.update_yaxes(showticklabels=False, row=row, col=1)
            fig.add_annotation(x=0, y=1.08, xref=f'x{row if row > 1 else ""} domain', yref=f'y{row if row > 1 else ""} domain', text=f'{col} — n={len(data):,}, all records retained', showarrow=False, xanchor='left', font=dict(size=12))
        fig.update_layout(title=f'Figure 7 — Extreme values flagged and retained ({int(unique_flags.sum())} records, 0 removed)', height=680)
        return fig

    # Figure 8 -------------------------------------------------------------
    if figure_number == 8:
        price = data['Median_Price'].dropna().astype(float)
        fig = make_subplots(rows=1, cols=2, subplot_titles=(f'Original scale — skew={price.skew():.2f}', f'Log display scale — log skew={np.log1p(price).skew():.2f}'))
        linear_edges = np.histogram_bin_edges(price, bins=60)
        log_edges = np.logspace(np.log10(price.min()), np.log10(price.max()), 45)
        fig.add_trace(_histogram_trace(price, linear_edges, 'Price', '#667085'), row=1, col=1)
        fig.add_trace(_histogram_trace(price, log_edges, 'Price', '#2F6FED'), row=1, col=2)
        fig.update_xaxes(title_text='Median price (RM)', row=1, col=1)
        fig.update_xaxes(title_text='Median price (RM, log display scale)', type='log', row=1, col=2)
        fig.update_yaxes(title_text='Records', row=1, col=1)
        fig.update_layout(title=f'Figure 8 — Price distribution with all {len(data):,} records retained', bargap=0.02)
        return fig

    # Figure 9 -------------------------------------------------------------
    if figure_number == 9:
        area_counts = data['Area_Key'].value_counts()
        band_order = ['Singleton Areas (1)', 'Low-frequency Areas (2-4)', 'Moderate-frequency Areas (5-9)', 'Well-represented Areas (10+)']
        bands = pd.cut(area_counts, bins=[0, 1, 4, 9, np.inf], labels=band_order)
        band_counts = bands.value_counts().reindex(band_order)
        fig = go.Figure(go.Pie(labels=band_counts.index, values=band_counts.values, hole=0.45, textinfo='percent', textposition='inside', marker=dict(colors=['#C63C4A', '#C47A10', '#2F6FED', '#18875D']), hovertemplate='%{label}<br>%{value:,} Areas<br>%{percent}<extra></extra>'))
        fig.update_layout(title='Figure 9 — Share of Areas by record-frequency band', annotations=[dict(text=f'{int(band_counts.sum()):,}<br>Areas', x=0.5, y=0.5, showarrow=False, font=dict(size=15))], legend_title_text='Area frequency band')
        return fig

    # Figure 10 ------------------------------------------------------------
    if figure_number == 10:
        counts = data['State'].value_counts().sort_values()
        fig = px.bar(x=counts.values, y=counts.index, orientation='h', text=counts.values, title='Figure 10 — Record count by State', labels={'x': 'Records', 'y': ''})
        fig.update_traces(marker_color='#2F6FED', textposition='outside', hovertemplate='%{y}<br>Records: %{x:,}<extra></extra>')
        fig.update_xaxes(range=[0, counts.max() * 1.12])
        return fig

    # Figure 11 ------------------------------------------------------------
    if figure_number == 11:
        counts = data['Primary_Type'].value_counts().sort_values()
        fig = px.bar(x=counts.values, y=counts.index, orientation='h', text=counts.values, title='Figure 11 — Record count by Primary_Type', labels={'x': 'Records', 'y': ''})
        fig.update_traces(marker_color='#18875D', textposition='outside', hovertemplate='%{y}<br>Records: %{x:,}<extra></extra>')
        fig.update_xaxes(range=[0, counts.max() * 1.12])
        return fig

    # Figure 12 ------------------------------------------------------------
    if figure_number == 12:
        counts = data['Category'].value_counts()
        fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, texttemplate='%{label}<br>%{value:,}<br>(%{percent})', textinfo='label+value+percent', marker=dict(colors=['#2F6FED', '#C47A10']), hovertemplate='%{label}<br>%{value:,} records<br>%{percent}<extra></extra>'))
        fig.update_layout(title='Figure 12 — Landed versus High-Rise composition', showlegend=False)
        return fig

    # Figures 13-18: exact filters/order from Latest_FINAL -----------------
    if figure_number == 13:
        state_n = data['State'].value_counts()
        major_states = state_n[state_n >= 10].index
        frame = data[data['State'].isin(major_states)].copy()
        order = frame.groupby('State')['Median_Price'].median().sort_values(ascending=False).index.tolist()
        frame['State label'] = frame['State'].map(lambda s: f'{s} (n={state_n[s]:,})')
        label_order = [f'{s} (n={state_n[s]:,})' for s in order]
        fig = px.box(frame, x='Median_Price', y='State label', points=False, category_orders={'State label': label_order}, title='Figure 13 — Price distribution by State (states with n ≥ 10)', labels={'Median_Price': 'Median price (RM)', 'State label': ''})
        fig.update_traces(fillcolor='#D5E4FF', line_color='#9CA3AF', quartilemethod='linear', hovertemplate='%{y}<br>Median price: RM %{x:,.0f}<extra></extra>')
        fig.update_yaxes(autorange='reversed')
        _apply_box_axis_range(fig, [(s, frame[frame['State'].eq(s)]) for s in order], 'Median_Price')
        return fig

    if figure_number == 14:
        min_area_n = 15
        top_areas = 10
        area_counts = data['Area_Key'].value_counts()
        area_keys = area_counts[area_counts >= min_area_n].head(top_areas).index
        frame = data[data['Area_Key'].isin(area_keys)].copy()
        frame['Area display'] = frame['Area_Key'].map(lambda v: display_name(str(v).replace(' | ', ' — ')))
        order = frame.groupby('Area display')['Median_Price'].median().sort_values(ascending=False).index.tolist()
        area_n = frame['Area display'].value_counts()
        frame['Area label'] = frame['Area display'].map(lambda a: f'{a} (n={area_n[a]})')
        label_order = [f'{a} (n={area_n[a]})' for a in order]
        fig = px.box(frame, x='Median_Price', y='Area label', points=False, category_orders={'Area label': label_order}, title=f'Figure 14 — Price distribution by Area (top {top_areas} Areas with n ≥ {min_area_n})', labels={'Median_Price': 'Median price (RM)', 'Area label': ''})
        fig.update_traces(fillcolor='#D7F1E5', line_color='#9CA3AF', quartilemethod='linear', hovertemplate='%{y}<br>Median price: RM %{x:,.0f}<extra></extra>')
        fig.update_yaxes(autorange='reversed')
        _apply_box_axis_range(fig, [(a, frame[frame['Area display'].eq(a)]) for a in order], 'Median_Price')
        return fig

    if figure_number == 15:
        counts = data['Primary_Type'].value_counts()
        major = counts[counts >= 10].index
        frame = data[data['Primary_Type'].isin(major)].copy()
        order = frame.groupby('Primary_Type')['Median_Price'].median().sort_values(ascending=False).index.tolist()
        frame['Type label'] = frame['Primary_Type'].map(lambda t: f'{t} (n={counts[t]:,})')
        label_order = [f'{t} (n={counts[t]:,})' for t in order]
        fig = px.box(frame, x='Median_Price', y='Type label', points=False, category_orders={'Type label': label_order}, title='Figure 15 — Price by property type (types with n ≥ 10)', labels={'Median_Price': 'Median price (RM)', 'Type label': ''})
        fig.update_traces(fillcolor='#FFE7C2', line_color='#9CA3AF', quartilemethod='linear', hovertemplate='%{y}<br>Median price: RM %{x:,.0f}<extra></extra>')
        fig.update_yaxes(autorange='reversed')
        _apply_box_axis_range(fig, [(t, frame[frame['Primary_Type'].eq(t)]) for t in order], 'Median_Price')
        return fig

    if figure_number == 16:
        counts = data['Tenure'].value_counts()
        order = data.groupby('Tenure')['Median_Price'].median().sort_values(ascending=False).index.tolist()
        frame = data.copy()
        frame['Tenure label'] = frame['Tenure'].map(lambda t: f'{t} (n={counts[t]:,})')
        label_order = [f'{t} (n={counts[t]:,})' for t in order]
        fig = px.box(frame, x='Median_Price', y='Tenure label', points=False, category_orders={'Tenure label': label_order}, title='Figure 16 — Price distribution by tenure', labels={'Median_Price': 'Median price (RM)', 'Tenure label': ''})
        fig.update_traces(fillcolor='#E4DCFB', line_color='#9CA3AF', quartilemethod='linear', hovertemplate='%{y}<br>Median price: RM %{x:,.0f}<extra></extra>')
        fig.update_yaxes(autorange='reversed')
        _apply_box_axis_range(fig, [(t, frame[frame['Tenure'].eq(t)]) for t in order], 'Median_Price')
        return fig

    if figure_number == 17:
        counts = data['Category'].value_counts()
        order = data.groupby('Category')['Median_Price'].median().sort_values(ascending=False).index.tolist()
        frame = data.copy()
        frame['Category label'] = frame['Category'].map(lambda c: f'{c} (n={counts[c]:,})')
        label_order = [f'{c} (n={counts[c]:,})' for c in order]
        fig = px.box(frame, x='Median_Price', y='Category label', points=False, category_orders={'Category label': label_order}, title='Figure 17 — Price by broad property category', labels={'Median_Price': 'Median price (RM)', 'Category label': ''})
        fig.update_traces(fillcolor='#CFE9E0', line_color='#9CA3AF', quartilemethod='linear', hovertemplate='%{y}<br>Median price: RM %{x:,.0f}<extra></extra>')
        fig.update_yaxes(autorange='reversed')
        _apply_box_axis_range(fig, [(c, frame[frame['Category'].eq(c)]) for c in order], 'Median_Price')
        return fig

    if figure_number == 18:
        state_n = data['State'].value_counts()
        major_states = state_n[state_n >= 10].index
        frame = data[data['State'].isin(major_states)].copy()
        order = frame.groupby('State')['Median_PSF'].median().sort_values(ascending=False).index.tolist()
        frame['State label'] = frame['State'].map(lambda s: f'{s} (n={state_n[s]:,})')
        label_order = [f'{s} (n={state_n[s]:,})' for s in order]
        fig = px.box(frame, x='Median_PSF', y='State label', points=False, category_orders={'State label': label_order}, title='Figure 18 — Median PSF distribution by State (states with n ≥ 10)', labels={'Median_PSF': 'Median price per square foot (RM)', 'State label': ''})
        fig.update_traces(fillcolor='#FBD7DC', line_color='#9CA3AF', quartilemethod='linear', hovertemplate='%{y}<br>Median PSF: RM %{x:,.0f}<extra></extra>')
        fig.update_yaxes(autorange='reversed')
        _apply_box_axis_range(fig, [(s, frame[frame['State'].eq(s)]) for s in order], 'Median_PSF')
        return fig

    # Figure 19 ------------------------------------------------------------
    if figure_number == 19:
        categories = ['Landed', 'High-Rise']
        colours = ['#2F6FED', '#C47A10']
        subplot_titles = []
        for category in categories:
            part = data[data['Category'].eq(category)]
            subplot_titles.append(f"{category} (n={len(part):,}, r={part['Median_PSF'].corr(part['Median_Price']):.2f})")
        fig = make_subplots(rows=1, cols=2, subplot_titles=subplot_titles, shared_xaxes=True, shared_yaxes=True)
        for col, (category, colour) in enumerate(zip(categories, colours), start=1):
            part = data[data['Category'].eq(category)].copy()
            fig.add_trace(go.Scatter(x=part['Median_PSF'], y=part['Median_Price'], mode='markers', marker=dict(size=6, color=colour, opacity=0.25), customdata=part[['State', 'Area_Clean', 'Primary_Type']], hovertemplate='PSF: RM %{x:,.0f}<br>Price: RM %{y:,.0f}<br>State: %{customdata[0]}<br>Area: %{customdata[1]}<br>Type: %{customdata[2]}<extra></extra>', showlegend=False), row=1, col=col)
            if len(part) > 1:
                slope, intercept = np.polyfit(part['Median_PSF'], part['Median_Price'], 1)
                xs = np.linspace(part['Median_PSF'].min(), part['Median_PSF'].max(), 100)
                fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode='lines', line=dict(color='#C63C4A', width=2), hoverinfo='skip', showlegend=False), row=1, col=col)
            fig.update_xaxes(title_text='Median PSF (RM)', row=1, col=col)
            fig.update_yaxes(type='log', row=1, col=col)
        fig.update_yaxes(title_text='Median price (RM, log display scale)', row=1, col=1)
        fig.update_layout(title='Figure 19 — Median PSF against Median Price, by category', height=520)
        return fig

    # Figure 20 ------------------------------------------------------------
    if figure_number == 20:
        x = data['Transactions'].astype(float)
        y = data['Median_Price'].astype(float)
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        pearson = x.corr(y)
        spearman = x.corr(y, method='spearman')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=6, color='#667085', opacity=0.25), customdata=data[['State', 'Area_Clean', 'Primary_Type']], hovertemplate='Transactions: %{x:,.0f}<br>Price: RM %{y:,.0f}<br>State: %{customdata[0]}<br>Area: %{customdata[1]}<br>Type: %{customdata[2]}<extra></extra>', name='Records'))
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode='lines', line=dict(color='#C63C4A', width=2), name='Linear trend'))
        fig.update_xaxes(title_text='Transactions')
        fig.update_yaxes(title_text='Median price (RM, log display scale)', type='log')
        fig.update_layout(title=f'Figure 20 — Transactions against Median Price (Pearson r={pearson:.3f}, Spearman={spearman:.3f})')
        return fig

    # Figure 21 ------------------------------------------------------------
    if figure_number == 21:
        encoded = pd.get_dummies(data[['Tenure', 'Primary_Type']], prefix=['Tenure', 'Primary_Type'], dtype=float)
        encoded = pd.concat([encoded.reset_index(drop=True), data[['Median_PSF', 'Transactions']].reset_index(drop=True)], axis=1)
        encoded['Median_Price'] = data['Median_Price'].values
        target_corr = encoded.corr(numeric_only=True)['Median_Price'].drop('Median_Price')
        tenure_cols = target_corr[[c for c in target_corr.index if c.startswith('Tenure_')]].abs().sort_values(ascending=False).index.tolist()
        type_cols = target_corr[[c for c in target_corr.index if c.startswith('Primary_Type_')]].abs().sort_values(ascending=False).index.tolist()
        ordered = ['Median_PSF', 'Transactions'] + tenure_cols + type_cols + ['Median_Price']
        corr = encoded[ordered].corr()
        z = corr.to_numpy(dtype=float)
        upper = np.triu(np.ones_like(z, dtype=bool), k=1)
        z[upper] = np.nan
        text_values = []
        for r in range(len(corr.index)):
            text_values.append([f'{corr.iloc[r, c]:.2f}' if (not upper[r, c] and abs(corr.iloc[r, c]) >= 0.10) else '' for c in range(len(corr.columns))])
        fig = go.Figure(go.Heatmap(z=z, x=corr.columns, y=corr.index, zmin=-1, zmax=1, zmid=0, colorscale='RdBu', reversescale=True, text=text_values, texttemplate='%{text}', hovertemplate='%{y} vs %{x}<br>r=%{z:.3f}<extra></extra>', colorbar=dict(title='r')))
        fig.update_layout(title='Figure 21 — Correlation matrix (lower triangle)<br><sup>Numeric + Tenure + Primary_Type; |r| ≥ 0.10 annotated</sup>', height=760)
        fig.update_xaxes(tickangle=45)
        return fig

    # Figure 22 ------------------------------------------------------------
    if figure_number == 22:
        association = pd.Series({'Median PSF': abs(data['Median_PSF'].corr(data['Median_Price'])), 'Transactions': abs(data['Transactions'].corr(data['Median_Price']))})
        for col, label in [('State', 'State'), ('Area_Key', 'Area'), ('Tenure', 'Tenure'), ('Primary_Type', 'Property type')]:
            association[label] = _live_correlation_ratio(data[col], data['Median_Price'])
        association = association.sort_values()
        fig = px.bar(x=association.values, y=association.index, orientation='h', text=[f'{v:.3f}' for v in association.values], title='Figure 22 — Association of each final model feature with price', labels={'x': 'Association strength with Median_Price', 'y': ''})
        fig.update_traces(marker_color='#2F6FED', textposition='outside', hovertemplate='%{y}<br>Association: %{x:.3f}<extra></extra>')
        fig.update_xaxes(title_text='Association strength with Median_Price<br>(|Pearson r| for numeric variables; correlation ratio η for categorical)', range=[0, association.max() * 1.18])
        return fig

    # Figure 23 ------------------------------------------------------------
    if figure_number == 23:
        min_cell_n = 10
        price_matrix = data.pivot_table(index='State', columns='Category', values='Median_Price', aggfunc='median')
        count_matrix = data.pivot_table(index='State', columns='Category', values='Median_Price', aggfunc='size')
        thin = count_matrix.isna() | (count_matrix < min_cell_n)
        price_matrix = price_matrix.loc[~thin.all(axis=1)]
        count_matrix = count_matrix.loc[price_matrix.index]
        thin = thin.loc[price_matrix.index]
        z = price_matrix.to_numpy(dtype=float)
        z[thin.to_numpy()] = np.nan
        text_values, hover_values = [], []
        for state in price_matrix.index:
            text_row, hover_row = [], []
            for category in price_matrix.columns:
                n = count_matrix.loc[state, category]
                if pd.isna(n) or n < min_cell_n:
                    text_row.append('—')
                    hover_row.append(f'{state} · {category}<br>Fewer than {min_cell_n} records')
                else:
                    value = price_matrix.loc[state, category]
                    text_row.append(f'RM{value/1000:,.0f}K<br>n={int(n)}')
                    hover_row.append(f'{state} · {category}<br>Median price: RM {value:,.0f}<br>n={int(n)}')
            text_values.append(text_row)
            hover_values.append(hover_row)
        fig = go.Figure(go.Heatmap(z=z, x=price_matrix.columns, y=price_matrix.index, colorscale='YlOrRd', text=text_values, texttemplate='%{text}', customdata=hover_values, hovertemplate='%{customdata}<extra></extra>', colorbar=dict(title='Median price (RM)')))
        fig.update_layout(title=f'Figure 23 — Median price by State × Category<br><sup>Cells with fewer than {min_cell_n} records are masked</sup>', height=690, plot_bgcolor='#E7E9EE')
        fig.update_xaxes(title_text='')
        fig.update_yaxes(title_text='')
        return fig

    # Figure 24 ------------------------------------------------------------
    if figure_number == 24:
        min_cell_n = 10
        price_matrix = data.pivot_table(index='Primary_Type', columns='Tenure', values='Median_Price', aggfunc='median')
        count_matrix = data.pivot_table(index='Primary_Type', columns='Tenure', values='Median_Price', aggfunc='size')
        thin = count_matrix.isna() | (count_matrix < min_cell_n)
        keep = ~thin.all(axis=1)
        price_matrix = price_matrix.loc[keep]
        count_matrix = count_matrix.loc[keep]
        thin = thin.loc[keep]
        z = price_matrix.to_numpy(dtype=float)
        z[thin.to_numpy()] = np.nan
        text_values, hover_values = [], []
        for ptype in price_matrix.index:
            text_row, hover_row = [], []
            for tenure in price_matrix.columns:
                n = count_matrix.loc[ptype, tenure]
                if pd.isna(n) or n < min_cell_n:
                    text_row.append('—')
                    hover_row.append(f'{ptype} · {tenure}<br>Fewer than {min_cell_n} records')
                else:
                    value = price_matrix.loc[ptype, tenure]
                    text_row.append(f'RM{value/1000:,.0f}K<br>n={int(n)}')
                    hover_row.append(f'{ptype} · {tenure}<br>Median price: RM {value:,.0f}<br>n={int(n)}')
            text_values.append(text_row)
            hover_values.append(hover_row)
        fig = go.Figure(go.Heatmap(z=z, x=price_matrix.columns, y=price_matrix.index, colorscale='YlGnBu', text=text_values, texttemplate='%{text}', customdata=hover_values, hovertemplate='%{customdata}<extra></extra>', colorbar=dict(title='Median price (RM)')))
        fig.update_layout(title=f'Figure 24 — Median price by Primary_Type × Tenure<br><sup>Cells with fewer than {min_cell_n} records are masked</sup>', height=600, plot_bgcolor='#E7E9EE')
        fig.update_xaxes(title_text='')
        fig.update_yaxes(title_text='')
        return fig

    return None

def _render_live_insight_card(figure_number, title, description, data, raw):
    st.markdown(
        f'<div class="mh-figure-card"><div class="mh-figure-title">{svg_icon("chart",18)} '
        f'Figure {figure_number} · {escape(title)}</div>'
        f'<div class="mh-figure-insight">{escape(description)}</div></div>',
        unsafe_allow_html=True,
    )
    figure = build_live_notebook_figure(figure_number, data, raw)
    if figure is not None:
        render_plotly(figure)
    else:
        st.info("Plotly is unavailable, so this interactive chart cannot be displayed.")


def insights_page(data):
    st.markdown(
        f'<div class="mh-hero"><div><h1>Market Insights</h1><p>Filter the 2025 housing dataset, '
        'compare market segments, and interact with the same analytical views used in Latest_FINAL.</p></div>'
        f'<div class="mh-hero-icon">{svg_icon("chart",34)}</div></div>', unsafe_allow_html=True,
    )
    view = st.radio("Insight view", ["Market Explorer", "Visual Insights"], horizontal=True,
                    label_visibility="collapsed", key="insights_view")

    if view == "Market Explorer":
        st.markdown(f'<div class="mh-panel-title">{svg_icon("filter",19)} Filter market records</div>', unsafe_allow_html=True)
        state_col, area_col, type_col, tenure_col = st.columns(4)
        state = state_col.selectbox("State", ["All"] + sorted(data["State"].dropna().unique()), key="explorer_state")
        state_pool = data if state == "All" else data[data["State"].eq(state)]
        areas = sorted(state_pool["Area_Clean"].dropna().unique())
        area_labels = {display_name(value): value for value in areas}
        area = area_col.selectbox("Area", ["All"] + list(area_labels), key="explorer_area")
        ptype = type_col.selectbox("Property type", ["All"] + sorted(data["Primary_Type"].dropna().unique()), key="explorer_type")
        tenure = tenure_col.selectbox("Tenure", ["All"] + sorted(data["Tenure"].dropna().unique()), key="explorer_tenure")

        subset = data.copy()
        if state != "All":
            subset = subset[subset["State"].eq(state)]
        if area != "All":
            subset = subset[subset["Area_Clean"].eq(area_labels[area])]
        if ptype != "All":
            subset = subset[subset["Primary_Type"].eq(ptype)]
        if tenure != "All":
            subset = subset[subset["Tenure"].eq(tenure)]

        if subset.empty:
            st.warning("No historical records match these filters.")
            return

        metric_cards([
            ("Records", f"{len(subset):,}", "Matching rows"),
            ("Median price", f"RM {subset['Median_Price'].median()/1000:,.0f}K", "Filtered market"),
            ("Median PSF", f"RM {subset['Median_PSF'].median():,.0f}", "Per square foot"),
            ("Median transactions", f"{subset['Transactions'].median():,.0f}", "Observed volume"),
        ])

        if HAS_PLOTLY:
            st.caption("These live views use the same chart logic and axes as Figures 13, 14, 15 and 19, but respond to the filters above.")

            left, right = st.columns(2)
            with left:
                counts = subset["State"].value_counts()
                valid_states = counts[counts >= 1].index
                frame = subset[subset["State"].isin(valid_states)].copy()
                order = frame.groupby("State")["Median_Price"].median().sort_values(ascending=False).index.tolist()
                frame["State label"] = frame["State"].map(lambda s: f"{s} (n={counts[s]:,})")
                fig = px.box(frame, x="Median_Price", y="State label", points=False,
                             category_orders={"State label": [f"{s} (n={counts[s]:,})" for s in order]},
                             title="Price distribution by State", labels={"Median_Price": "Median price (RM)", "State label": ""})
                fig.update_yaxes(autorange="reversed")
                render_plotly(fig)
                st.caption("Shows the spread of median prices for the currently filtered states.")

            with right:
                counts = subset["Primary_Type"].value_counts()
                order = subset.groupby("Primary_Type")["Median_Price"].median().sort_values(ascending=False).index.tolist()
                frame = subset.copy()
                frame["Type label"] = frame["Primary_Type"].map(lambda t: f"{t} (n={counts[t]:,})")
                fig = px.box(frame, x="Median_Price", y="Type label", points=False,
                             category_orders={"Type label": [f"{t} (n={counts[t]:,})" for t in order]},
                             title="Price distribution by Property Type", labels={"Median_Price": "Median price (RM)", "Type label": ""})
                fig.update_yaxes(autorange="reversed")
                render_plotly(fig)
                st.caption("Compares price distributions across the property types left by the filters.")

            left, right = st.columns(2)
            with left:
                fig = px.scatter(subset, x="Median_PSF", y="Median_Price", color="Category",
                                 hover_data=["State", "Area_Clean", "Primary_Type"],
                                 title="Median PSF against Median Price",
                                 labels={"Median_PSF": "Median PSF (RM)", "Median_Price": "Median price (RM, log display scale)"})
                fig.update_yaxes(type="log")
                render_plotly(fig)
                st.caption("Hover over a point to inspect how PSF and price move together for the filtered records.")

            with right:
                min_area_n = 15
                area_counts = subset["Area_Key"].value_counts()
                area_keys = area_counts[area_counts >= min_area_n].head(10).index
                if len(area_keys):
                    frame = subset[subset["Area_Key"].isin(area_keys)].copy()
                    frame["Area display"] = frame["Area_Key"].map(lambda v: display_name(str(v).replace(" | ", " — ")))
                    area_n = frame["Area display"].value_counts()
                    order = frame.groupby("Area display")["Median_Price"].median().sort_values(ascending=False).index.tolist()
                    frame["Area label"] = frame["Area display"].map(lambda a: f"{a} (n={area_n[a]})")
                    fig = px.box(frame, x="Median_Price", y="Area label", points=False,
                                 category_orders={"Area label": [f"{a} (n={area_n[a]})" for a in order]},
                                 title="Price distribution by Area (n >= 15)", labels={"Median_Price": "Median price (RM)", "Area label": ""})
                    fig.update_yaxes(autorange="reversed")
                    render_plotly(fig)
                    st.caption("Matches the Area comparison logic in Figure 14 and only shows better-represented Areas.")
                else:
                    st.info("The current filters do not leave any Area with at least 15 records for the Figure 14-style comparison.")
        else:
            st.info("Plotly is unavailable, so interactive Market Explorer charts cannot be displayed.")

        export_columns = [c for c in ["Township", "Area_Clean", "State", "Primary_Type", "Tenure", "Median_Price", "Median_PSF", "Transactions"] if c in subset.columns]
        with st.expander("View filtered records"):
            st.dataframe(subset[export_columns], use_container_width=True, hide_index=True)
        st.download_button(
            "Download filtered records (CSV)",
            data=subset[export_columns].to_csv(index=False).encode("utf-8"),
            file_name="malaysia_housing_filtered.csv",
            mime="text/csv",
            key="download_filtered",
        )
        return

    raw = load_raw_insight_data()
    category = st.selectbox("Insight category", list(LIVE_INSIGHT_GROUPS), key="insight_category")
    st.markdown(
        f'<div class="mh-takeaway"><strong>Key takeaway from this category</strong><br>'
        f'{escape(LIVE_INSIGHT_TAKEAWAYS[category])}</div>',
        unsafe_allow_html=True,
    )
    st.caption("All charts below are live Plotly versions of Figures 1–24 in Latest_FINAL. Hover, zoom, pan and use the chart toolbar during presentation.")

    for figure_number, title, description in LIVE_INSIGHT_GROUPS[category]:
        _render_live_insight_card(figure_number, title, description, data, raw)

MODEL_FIGURE_DESCRIPTIONS = {
    25: 'Compares Group CV RMSE with final test RMSE for all four models.',
    26: 'Shows how RMSE changes across the four unseen-Area validation folds.',
    27: 'Compares training and test RMSE to make overfitting easy to see.',
    28: 'Shows how much validation error changes when Median_PSF is removed.',
    29: 'Checks predicted-versus-actual values and the residual pattern of the selected model.',
    30: 'Shows how much test R² falls when each input is shuffled.',
    31: 'Shows the selected model’s native importance grouped back to the six original inputs.',
}


def build_live_model_figure(figure_number, results):
    if not HAS_PLOTLY:
        return None

    import numpy as np
    from plotly.subplots import make_subplots

    selected_name = selected_model_name(results)
    selected_mask = results['Model'].eq(selected_name)

    if figure_number == 25:
        ordered_cv = results.sort_values('Group_CV_RMSE_mean')
        ordered_test = results.sort_values('RMSE_test')
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Training-set cross-validation', 'Test set'), horizontal_spacing=0.16)
        cv_colors = ['#18875D' if m == selected_name else '#2F6FED' for m in ordered_cv['Model']]
        test_colors = ['#18875D' if m == selected_name else '#2F6FED' for m in ordered_test['Model']]
        fig.add_trace(go.Bar(x=ordered_cv['Group_CV_RMSE_mean']/1000, y=ordered_cv['Model'], orientation='h', marker_color=cv_colors, error_x=dict(type='data', array=ordered_cv['Group_CV_RMSE_std']/1000, visible=True, color='#667085', thickness=1.2), text=[f'RM {v/1000:,.1f}K' for v in ordered_cv['Group_CV_RMSE_mean']], textposition='outside', hovertemplate='%{y}<br>Group CV RMSE: RM %{x:,.1f}K<extra></extra>', showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=ordered_test['RMSE_test']/1000, y=ordered_test['Model'], orientation='h', marker_color=test_colors, text=[f'RM {v/1000:,.1f}K' for v in ordered_test['RMSE_test']], textposition='outside', hovertemplate='%{y}<br>Test RMSE: RM %{x:,.1f}K<extra></extra>', showlegend=False), row=1, col=2)
        fig.update_xaxes(title_text="Group CV RMSE (RM '000), lower is better", row=1, col=1)
        fig.update_xaxes(title_text="Test RMSE (RM '000), lower is better", row=1, col=2)
        fig.update_layout(title='Figure 25 — Model RMSE comparison', height=520)
        return fig

    if figure_number == 26:
        path = APP_DIR / 'fold_scores.csv'
        if not path.exists():
            return None
        folds = pd.read_csv(path)
        fig = go.Figure()
        for model_name in folds['Model'].drop_duplicates():
            part = folds[folds['Model'].eq(model_name)]
            fig.add_trace(go.Scatter(x=part['Fold'], y=part['RMSE (RM)']/1000, mode='lines+markers', name=model_name, line=dict(width=3 if model_name == selected_name else 1.8), marker=dict(size=8 if model_name == selected_name else 6), hovertemplate=f'{model_name}<br>Fold %{{x}}<br>RMSE: RM %{{y:,.1f}}K<extra></extra>'))
        fig.update_xaxes(title_text='Group CV fold', dtick=1)
        fig.update_yaxes(title_text="RMSE (RM '000), lower is better")
        fig.update_layout(title='Figure 26 — Group CV RMSE by fold', height=520, legend_title_text='Model')
        return fig

    if figure_number == 27:
        ordered = results.sort_values('RMSE_test')
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ordered['Model'], y=ordered['RMSE_train']/1000, name='Training RMSE', text=[f'{v/1000:,.1f}K' for v in ordered['RMSE_train']], textposition='outside', hovertemplate='%{x}<br>Training RMSE: RM %{y:,.1f}K<extra></extra>'))
        fig.add_trace(go.Bar(x=ordered['Model'], y=ordered['RMSE_test']/1000, name='Test RMSE', text=[f'{v/1000:,.1f}K' for v in ordered['RMSE_test']], textposition='outside', hovertemplate='%{x}<br>Test RMSE: RM %{y:,.1f}K<extra></extra>'))
        fig.update_yaxes(title_text="RMSE (RM '000), lower is better")
        fig.update_xaxes(tickangle=25)
        fig.update_layout(title='Figure 27 — Training versus test RMSE', barmode='group', height=520)
        return fig

    if figure_number == 28:
        path = APP_DIR / 'median_psf_ablation_results.csv'
        if not path.exists():
            return None
        check = pd.read_csv(path)
        fig = go.Figure(go.Bar(x=check['Feature set'], y=check['Validation RMSE (RM)']/1000, text=[f'RM {v/1000:,.1f}K' for v in check['Validation RMSE (RM)']], textposition='outside', hovertemplate='%{x}<br>Validation RMSE: RM %{y:,.1f}K<extra></extra>'))
        fig.update_yaxes(title_text="Validation RMSE (RM '000), lower is better")
        fig.update_layout(title=f'Figure 28 — Median_PSF dependency ({selected_name})', height=500)
        return fig

    if figure_number == 29:
        path = APP_DIR / 'test_predictions.csv'
        if not path.exists():
            return None
        preds = pd.read_csv(path)
        preds = preds.loc[:, ~preds.columns.str.startswith('Unnamed:')]
        if selected_name not in preds.columns or 'Actual' not in preds.columns:
            return None
        actual = preds['Actual'].astype(float)
        predicted = preds[selected_name].astype(float)
        residual = actual - predicted
        limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Predicted versus actual', 'Residual pattern'), horizontal_spacing=0.13)
        fig.add_trace(go.Scatter(x=actual, y=predicted, mode='markers', marker=dict(size=6, opacity=0.5, color='#2F6FED'), hovertemplate='Actual: RM %{x:,.0f}<br>Predicted: RM %{y:,.0f}<extra></extra>', showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=limits, y=limits, mode='lines', line=dict(dash='dash', width=2, color='#667085'), hoverinfo='skip', showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=predicted, y=residual, mode='markers', marker=dict(size=6, opacity=0.5, color='#2F6FED'), hovertemplate='Predicted: RM %{x:,.0f}<br>Residual: RM %{y:,.0f}<extra></extra>', showlegend=False), row=1, col=2)
        fig.add_hline(y=0, line_dash='dash', line_width=2, line_color='#667085', row=1, col=2)
        fig.update_xaxes(title_text='Actual median price (RM)', row=1, col=1)
        fig.update_yaxes(title_text='Predicted median price (RM)', row=1, col=1)
        fig.update_xaxes(title_text='Predicted median price (RM)', row=1, col=2)
        fig.update_yaxes(title_text='Residual (RM)', row=1, col=2)
        fig.update_layout(title=f'Figure 29 — Prediction diagnostics for {selected_name}', height=520)
        return fig

    if figure_number == 30:
        table = pd.DataFrame({
            'Feature': ['State', 'Transactions', 'Tenure', 'Area', 'Primary_Type', 'Median_PSF'],
            'Mean drop in R²': [-0.0015, -0.0011, -0.0009, 0.0000, 0.3233, 1.8032],
            'Standard deviation': [0.0026, 0.0121, 0.0037, 0.0000, 0.0391, 0.1056],
        })
        fig = go.Figure(go.Bar(x=table['Mean drop in R²'], y=table['Feature'], orientation='h', error_x=dict(type='data', array=table['Standard deviation'], visible=True, color='#667085', thickness=1.2), text=[f'{v:.4f}' for v in table['Mean drop in R²']], textposition='outside', hovertemplate='%{y}<br>Mean drop in R²: %{x:.4f}<extra></extra>'))
        fig.update_xaxes(title_text='Mean decrease in test R² when shuffled')
        fig.update_layout(title=f'Figure 30 — Test-set permutation importance ({selected_name})', height=520)
        return fig

    if figure_number == 31:
        table = pd.DataFrame({
            'Feature': ['Tenure', 'State', 'Transactions', 'Area', 'Primary Type', 'Median PSF'],
            'Share (%)': [0.4678, 2.0985, 6.2133, 7.5356, 16.9120, 66.7729],
        })
        fig = go.Figure(go.Bar(x=table['Share (%)'], y=table['Feature'], orientation='h', text=[f'{v:.1f}%' for v in table['Share (%)']], textposition='outside', hovertemplate='%{y}<br>Share: %{x:.1f}%<extra></extra>'))
        fig.update_xaxes(title_text=f'Share of {selected_name} native importance (%)')
        fig.update_layout(title=f'Figure 31 — Aggregated native importance ({selected_name})', height=520)
        return fig

    return None


# ---------------------------------------------------------------------------
# PAGE 3 - MODEL REPORT
# ---------------------------------------------------------------------------
def model_report_page(results):
    recommended_name = selected_model_name(results)
    recommended = results.loc[results["Model"].eq(recommended_name)].iloc[0]
    st.markdown(
        f'<div class="mh-model-hero"><span class="mh-badge">Recommended model</span>'
        f'<div class="mh-model-name">{escape(recommended_name)}</div>'
        '<div class="mh-help">Recommended from unseen-area Group CV performance and stability, then evaluated '
        'on a held-out test split. This prioritises generalisation to locations the model has not memorised.</div></div>',
        unsafe_allow_html=True,
    )
    metric_cards([
        ("Best CV RMSE", f"RM {recommended['Group_CV_RMSE_mean']/1000:,.1f}K", "Lower is better"),
        ("Test RMSE", f"RM {recommended['RMSE_test']/1000:,.1f}K", "Penalises large misses"),
        ("Test MAE", f"RM {recommended['MAE_test']/1000:,.1f}K", "Typical absolute error"),
        ("Test R²", f"{recommended['R2_test']:.3f}", "Higher is better"),
    ])
    st.markdown("### Model leaderboard")
    leaderboard = results.copy()
    leaderboard.insert(0, "Rank", range(1, len(leaderboard) + 1))
    leaderboard["Recommended"] = leaderboard["Model"].eq(recommended_name).map({True: "Best", False: ""})
    leaderboard = leaderboard[["Rank", "Model", "Recommended", "Group_CV_RMSE_mean", "RMSE_test", "MAE_test", "R2_test"]]
    st.dataframe(
        leaderboard.style.apply(lambda row: ["background-color:#EFF6FF;font-weight:700" if row["Recommended"] == "Best" else "" for _ in row], axis=1)
        .format({"Group_CV_RMSE_mean":"RM {:,.0f}", "RMSE_test":"RM {:,.0f}", "MAE_test":"RM {:,.0f}", "R2_test":"{:.3f}"}),
        use_container_width=True, hide_index=True,
    )
    st.markdown("### Interactive model comparison")
    st.caption("Lower RMSE and MAE indicate smaller errors. Higher R² indicates more explained price variation.")
    if HAS_PLOTLY:
        long = results.melt(id_vars="Model", value_vars=["Group_CV_RMSE_mean", "MAE_test"],
                            var_name="Metric", value_name="RM")
        long["Metric"] = long["Metric"].map({"Group_CV_RMSE_mean":"Group CV RMSE", "MAE_test":"Test MAE"})
        render_plotly(px.bar(long, x="Model", y="RM", color="Metric", barmode="group",
            title="Error comparison", color_discrete_map={"Group CV RMSE":"#2563EB", "Test MAE":"#0D9488"}))
        render_plotly(px.bar(results, x="Model", y="R2_test", title="Test R² comparison",
            color="R2_test", color_continuous_scale=["#DBEAFE", "#2563EB"], labels={"R2_test":"Test R²"}))
    else:
        st.bar_chart(results.set_index("Model")[["Group_CV_RMSE_mean", "MAE_test"]])
        st.bar_chart(results.set_index("Model")[["R2_test"]])
    with st.expander("Metric explanation"):
        st.markdown(
            "- **RMSE:** average error size that penalises large errors more heavily.\n"
            "- **MAE:** average absolute error; this is easier to interpret directly in RM.\n"
            "- **R²:** the share of price variation explained by the model; higher is better."
        )
    sections = {
        "Performance": [(25, "Cross-validation and test RMSE"),
                        (26, "Unseen-area fold stability"),
                        (27, "Train-test overfitting check")],
        "Diagnostics and dependency": [(28, "Median PSF dependency"),
                                       (29, "Prediction diagnostics")],
        "Importance": [(30, "Permutation importance"),
                       (31, "Grouped native importance")],
    }
    section = st.selectbox("Diagnostic section", list(sections), key="model_report_section")
    for figure_number, title in sections[section]:
        description = MODEL_FIGURE_DESCRIPTIONS.get(figure_number, "")
        st.markdown(
            f'<div class="mh-figure-card"><div class="mh-figure-title">{svg_icon("model",18)} Figure {figure_number} · {escape(title)}</div>'
            f'<div class="mh-figure-insight">{escape(description)}</div></div>',
            unsafe_allow_html=True,
        )
        figure = build_live_model_figure(figure_number, results)
        if figure is not None:
            render_plotly(figure)
        else:
            st.warning(f"Interactive Figure {figure_number} could not be generated because a required evaluation file is missing.")
    st.markdown(
        f'<div class="mh-limitations">{svg_icon("warning",20)} <strong>Limitations</strong><br>'
        'The app uses a 2025 dataset only; some areas are unseen or infrequent; the estimate depends heavily '
        'on an independently known PSF input; and the output is not a professional valuation.</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    results_path = resolve_results_path()
    missing = [path.name for path in [DATA_PATH, results_path] if not path.exists()]
    if missing:
        st.error("Missing required files: " + ", ".join(missing))
        st.stop()
    try:
        data = load_data()
        results = load_results()
    except Exception as error:
        st.error(f"Unable to load the housing data or model results: {error}")
        st.stop()
    required_data_columns = set(MODEL_FEATURES + ["Median_Price", "Area_Clean"])
    missing_columns = sorted(required_data_columns - set(data.columns))
    if missing_columns:
        st.error("Housing data is missing required columns: " + ", ".join(missing_columns))
        st.stop()
    recommended = selected_model_name(results)
    recommended_model_path = MODELS_DIR / (recommended.split(" (")[0].lower().replace(" ", "_") + ".pkl")
    if not recommended_model_path.exists():
        st.error(f"Missing selected model file: {recommended_model_path.name}")
        st.stop()
    coverage = map_coverage_summary()
    if coverage["dataset_pairs_missing_from_map"]:
        st.error("Internal map coverage check failed: dataset locations are missing from the nationwide map.")
        st.stop()
    missing_state_coords = sorted(set(coverage["states"]) - set(STATE_COORDS))
    if missing_state_coords:
        st.error("Missing map coordinates for: " + ", ".join(missing_state_coords))
        st.stop()
    prediction_tab, insights_tab, report_tab = st.tabs(["Price Prediction", "Market Insights", "Model Report"])
    with prediction_tab:
        prediction_page(data, results)
    with insights_tab:
        insights_page(data)
    with report_tab:
        model_report_page(results)
    st.markdown(
        f'<div class="mh-footer"><div><div class="brand">{svg_icon("home",19)} Malaysia Housing Estimator</div>'
        '<div class="sub">Nationwide map · validated model and evaluation outputs</div></div>'
        '<div class="sub">BMDS2003 · Data Science Prototype</div></div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
