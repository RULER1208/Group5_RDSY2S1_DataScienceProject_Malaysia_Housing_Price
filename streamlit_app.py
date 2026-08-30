"""
BMDS2003 Data Science - Deployment Prototype
Malaysia Housing Median Price Estimator

Run locally:  streamlit run streamlit_app.py
"""
from __future__ import annotations
from pathlib import Path
from html import escape
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
    --navy:#17233B;
    --navy-2:#223452;
    --navy-3:#2D4264;
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

/* ---------- OLD UI DARK HEADER / NAVIGATION ---------- */
.stTabs [role="tablist"] {
    display:flex!important;
    align-items:center!important;
    gap:8px!important;
    min-height:82px;
    padding:14px 18px!important;
    background:linear-gradient(135deg, #18243B 0%, #22324E 100%)!important;
    border:1px solid rgba(255,255,255,.07)!important;
    border-radius:21px!important;
    margin:10px 0 28px!important;
    box-shadow:0 14px 30px rgba(23,35,59,.16);
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
    border-right:1px solid rgba(255,255,255,.14);
    font-size:1.08rem;
    font-weight:800;
    letter-spacing:.01em;
    color:#FFFFFF;
}
.stTabs [role="tab"] {
    align-self:stretch!important;
    height:auto!important;
    padding:0 24px!important;
    border-radius:12px!important;
    color:#C5D0E2!important;
    font-weight:650!important;
    font-size:.94rem!important;
    background:transparent!important;
    border:1px solid transparent!important;
    transition:all .16s ease!important;
    cursor:pointer!important;
}
.stTabs [role="tab"]:hover { color:#FFFFFF!important; background:rgba(255,255,255,.065)!important; }
.stTabs [role="tab"][aria-selected="true"] {
    color:#1E2B43!important;
    background:#F8FAFC!important;
    border-color:#DDE4EE!important;
    box-shadow:0 6px 14px rgba(8,18,36,.15)!important;
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
.mh-map-legend .state-dot { background:#2F6FED; border:2px solid #15243A; }
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
.mh-metric { background:linear-gradient(135deg, #17233B 0%, #22324E 100%); border-radius:16px;
    padding:15px 17px; border:1px solid rgba(255,255,255,.08); }
.mh-metric .k { font-family:var(--mono); font-size:.65rem; letter-spacing:.12em; color:#B8C5DD; text-transform:uppercase; }
.mh-metric .v { font-family:var(--mono); font-size:1.28rem; font-weight:800; color:#FFFFFF; margin-top:6px; }
.mh-metric .hint { color:#BBC7DB; font-size:.73rem; margin-top:4px; }
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
    background:linear-gradient(135deg, #18243B 0%, #22324E 100%); border:1px solid rgba(255,255,255,.06);
    color:#DCE4F2; display:flex; align-items:center; justify-content:space-between; gap:18px;
    box-shadow:0 13px 30px rgba(23,35,59,.14); border-top:3px solid #7188C8; }
.mh-footer .brand { color:#FFFFFF; font-weight:800; letter-spacing:.01em; }
.mh-footer .sub { color:#BBC7DB; font-size:.84rem; }

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

@st.cache_data(show_spinner=False)
def get_area_coords(area_name: str, state_name: str):
    """Return a verified precomputed coordinate without network calls."""
    if state_name in HARDCODED_AREAS and area_name in HARDCODED_AREAS[state_name]:
        return HARDCODED_AREAS[state_name][area_name]
    if area_name == state_name:
        return STATE_COORDS.get(state_name)
    return None

@st.cache_data(show_spinner=False)
def get_area_map_coords(area_name: str, state_name: str):
    """Always return a map pin while preserving whether its position is verified."""
    real = get_area_coords(area_name, state_name)
    if real:
        return real, False

    # If live geocoding is unavailable, keep the area selectable but mark the pin
    # as approximate rather than pretending the fallback coordinate is exact.
    base = STATE_COORDS.get(state_name)
    if not base:
        return None, True
    h = int(hashlib.md5(f"{state_name}|{area_name}".encode("utf-8")).hexdigest(), 16)
    lat_offset = (((h % 1000) / 999) - 0.5) * 0.55
    lon_offset = ((((h // 1000) % 1000) / 999) - 0.5) * 0.75
    return [base[0] + lat_offset, base[1] + lon_offset], True

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


@st.cache_data(show_spinner=False)
def get_area_marker_data(state_name: str):
    """Cache the small marker table for one selected state."""
    marker_rows = []
    for area_name in get_areas_for_state(state_name):
        coordinates, approximate = get_area_map_coords(area_name, state_name)
        if coordinates:
            marker_rows.append((area_name, coordinates[0], coordinates[1], approximate, get_area_source(state_name, area_name)))
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
def geocode_malaysia_location(query: str):
    if not HAS_GEOPY or not query:
        return None
    try:
        geolocator = Nominatim(user_agent="malaysia_housing_estimator_student", timeout=2)
        loc = geolocator.geocode(f"{query}, Malaysia", country_codes="my", addressdetails=True, exactly_one=True, limit=1)
        if not loc:
            return None
        raw = getattr(loc, "raw", {}) or {}
        details = raw.get("address", {}) or {}
        return {
            "lat": float(loc.latitude),
            "lon": float(loc.longitude),
            "display_name": raw.get("display_name") or query,
            "state_raw": details.get("state") or details.get("region") or "",
            "area_raw": (
                details.get("suburb") or details.get("town") or details.get("city") or
                details.get("municipality") or details.get("county") or details.get("district") or
                details.get("village") or details.get("neighbourhood") or ""
            ),
        }
    except Exception:
        return None


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


def analyze_address(available_states):
    addr_raw = st.session_state.get("address_input", "")
    addr_norm = normalise_lookup_text(addr_raw)
    if not addr_norm:
        st.session_state["address_feedback"] = None
        return

    matched_state = None
    matched_area = None
    match_source = None
    searched_point = None

    # Fast offline signals first: area text, state text and postcode ranges.
    postcode_match = re.search(r"\b\d{5}\b", str(addr_raw))
    postcode_state = get_state_from_postcode(postcode_match.group()) if postcode_match else None
    text_state = match_state_text(addr_norm)
    preferred_states = [state for state in [postcode_state, text_state] if state in available_states]
    area_state, area_name = match_area_text(addr_norm, preferred_states=preferred_states)

    if area_state and area_name:
        matched_state = area_state
        matched_area = area_name
        match_source = "locality"
    elif postcode_state:
        matched_state = postcode_state
        match_source = "postcode"
    elif text_state:
        matched_state = text_state
        match_source = "state"

    # Optional live geocoding is only used after clicking Search Location.
    # It is cached, Malaysia-restricted, and never runs during map rendering.
    should_geocode = HAS_GEOPY and (not matched_area or len(addr_norm.split()) >= 3)
    if should_geocode:
        geo = geocode_malaysia_location(addr_raw)
        if geo:
            searched_point = {"lat": geo["lat"], "lon": geo["lon"], "label": geo["display_name"]}
            geo_state_norm = normalise_lookup_text(geo.get("state_raw", ""))
            geo_state = None
            for state_name in available_states:
                if contains_lookup_phrase(geo_state_norm, state_name) or contains_lookup_phrase(state_name, geo_state_norm):
                    geo_state = state_name
                    break
            if not geo_state:
                geo_state = match_state_text(geo_state_norm)

            if geo_state:
                # Geocoded state improves postcode-only matches, but a strong local
                # area match such as Setapak still keeps its known state.
                if not matched_area:
                    matched_state = geo_state
                    match_source = "geocode"
                elif matched_state == postcode_state and text_state and text_state != postcode_state and area_state != postcode_state:
                    matched_state = geo_state

            if geo.get("area_raw"):
                raw_area_norm = normalise_lookup_text(geo["area_raw"])
                preferred = [state for state in [matched_state, geo_state, postcode_state, text_state] if state]
                geo_area_state, geo_area = match_area_text(raw_area_norm, preferred_states=preferred)
                if geo_area_state and geo_area and not matched_area:
                    matched_state, matched_area = geo_area_state, geo_area
                    match_source = "geocode"
                elif not matched_area and matched_state:
                    # Keep Malaysia-wide coverage even when the exact place is not in
                    # the housing dataset or official list. The model can still handle
                    # this as an unseen/infrequent area.
                    matched_area = display_name(clean_area_name(geo["area_raw"]))
                    match_source = "geocode-custom"

    if matched_state:
        st.session_state["last_prediction"] = None
        st.session_state["selected_state"] = matched_state
        st.session_state["searched_point"] = searched_point
        if matched_area:
            st.session_state["selected_area"] = matched_area
            if searched_point:
                st.session_state["map_center"] = [searched_point["lat"], searched_point["lon"]]
                st.session_state["map_zoom"] = 14
            else:
                coords, _ = get_area_map_coords(matched_area, matched_state)
                if coords:
                    st.session_state["map_center"] = coords
                    st.session_state["map_zoom"] = 12
            st.session_state["address_feedback"] = ("success", f"Matched **{matched_area}, {matched_state}**.")
        else:
            st.session_state["selected_area"] = None
            if searched_point:
                st.session_state["map_center"] = [searched_point["lat"], searched_point["lon"]]
                st.session_state["map_zoom"] = 13
            else:
                st.session_state["map_center"] = STATE_COORDS.get(matched_state, [4.2105, 108.9758])
                st.session_state["map_zoom"] = 8
            st.session_state["address_feedback"] = (
                "info", f"Matched state **{matched_state}**. Pick an area on the map below."
            )
    else:
        st.session_state["searched_point"] = None
        st.session_state["address_feedback"] = (
            "warning",
            "Couldn't recognise that address or postcode. Try a postcode, township name, or select the location on the map.",
        )

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


def render_result(saved):
    lower = max(0.0, saved["prediction"] - saved["mae_test"])
    upper = saved["prediction"] + saved["mae_test"]
    warning = ""
    if not saved["dataset_supported"]:
        warning = (
            f'<div class="mh-warning">{svg_icon("warning", 20)}<div><strong>Lower-confidence area.</strong> '
            'This area is outside the 2025 training dataset. The model uses unseen/infrequent-area '
            'handling, so accuracy may be lower.</div></div>'
        )
    elif not saved["model_supported"]:
        warning = (
            f'<div class="mh-warning">{svg_icon("info", 20)}<div>This area exists in the dataset but '
            'was held outside this model’s training areas during evaluation. Unseen-area handling is used.</div></div>'
        )
    html = (
        '<div class="mh-result" id="estimate-result"><div class="mh-result-top"><div>'
        '<div class="cap">Estimated median price</div>'
        f'<div class="price">RM {saved["prediction"]:,.0f}</div>'
        f'<div class="sub">{escape(saved["area"])}, {escape(saved["state"])} · '
        f'{escape(saved["ptype"])} · {escape(saved["tenure"])}</div>'
        f'<div class="mh-range">Expected range: RM {lower/1000:,.0f}K – RM {upper/1000:,.0f}K</div>'
        '</div><div class="mh-result-badge"><div class="kicker">Selected model</div>'
        f'<div class="model">{escape(saved["model_name"])}</div></div></div>'
        '<div class="mh-stats">'
        f'<div class="mh-stat"><div class="k">Location</div><div class="v">{escape(saved["state"])}</div></div>'
        f'<div class="mh-stat"><div class="k">Median PSF input</div><div class="v">RM {saved["psf"]:,.0f}</div></div>'
        f'<div class="mh-stat"><div class="k">Test MAE</div><div class="v">RM {saved["mae_test"]/1000:,.1f}K</div></div>'
        f'<div class="mh-stat"><div class="k">Test R²</div><div class="v">{saved["r2_test"]:.3f}</div></div>'
        f'</div>{warning}<div class="mh-disclaimer">This is an estimated median price, not a final market valuation.</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


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
        "saved_scenarios": [],
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
            placeholder="e.g. 45400 or Sekinchan, Selangor",
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

    if HAS_INTERACTIVE_MAP:
        map_center = st.session_state.get(
            "map_center",
            [4.2105, 108.9758] if not current_state else STATE_COORDS.get(current_state, [4.2105, 108.9758]),
        )
        map_zoom = st.session_state.get("map_zoom", 6 if not current_state else 9)
        malaysia_map = folium.Map(
            location=map_center,
            zoom_start=map_zoom,
            tiles="OpenStreetMap",
            control_scale=True,
            zoom_control=True,
            prefer_canvas=True,
        )

        if not current_state:
            for state_name in available_states:
                folium.CircleMarker(
                    location=STATE_COORDS[state_name],
                    radius=11,
                    color="#15243A",
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

            for disp_area, lat, lon, is_approx, area_kind in marker_rows:
                is_sel = disp_area == current_area
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

            searched_point = st.session_state.get("searched_point")
            if searched_point and current_state:
                folium.CircleMarker(
                    location=[searched_point["lat"], searched_point["lon"]],
                    radius=10,
                    color="#0F766E",
                    weight=3,
                    fill=True,
                    fill_color="#0D9488",
                    fill_opacity=0.95,
                    tooltip=folium.Tooltip(f"<b>Searched address</b><br>{escape(str(searched_point.get('label', '')))}", sticky=True),
                    popup="SEARCHED_POINT",
                ).add_to(malaysia_map)

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
            "<span class='legend-item'><span class='legend-dot verified-dot'></span>Verified area</span>"
            "<span class='legend-item'><span class='legend-dot approx-dot'></span>Approximate position</span>"
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
                    st.session_state["pending_location"] = (clicked_st, None)
                    st.session_state["map_center"] = STATE_COORDS.get(clicked_st, [4.2105, 108.9758])
                    st.session_state["map_zoom"] = 9
                    st.session_state["searched_point"] = None
                    clear_last_prediction()
                    st.rerun()
            elif popup_txt.startswith("AREA:"):
                clicked_area = popup_txt.split(":", 1)[1]
                if clicked_area != current_area:
                    st.session_state["pending_location"] = (current_state, clicked_area)
                    coords, _ = get_area_map_coords(clicked_area, current_state)
                    if coords:
                        st.session_state["map_center"] = coords
                    st.session_state["map_zoom"] = 12
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
        "<div class='mh-help'>ⓘ Verified areas come from the dataset or known coordinates. "
        "Approximate and custom searched locations remain selectable for nationwide Malaysia prediction, "
        "but the result may have lower confidence when the area is not in the training dataset.</div>",
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
    st.caption(f"PSF reference: RM {reference['psf']:,}/sq ft from the {reference['label']} ({reference['rows']:,} record(s)).")

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

    saved = st.session_state.get("last_prediction")
    if saved:
        render_result(saved)
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
FIGURE_GROUPS = {
    "Data quality": [
        ("fig01_raw_target_distribution.png", "Raw target distribution", "Median Price is right-skewed, with a small premium segment."),
        ("fig02_raw_numeric_boxplots.png", "Numeric range and outliers", "The numerical features include genuine extreme market values."),
        ("fig03_raw_categories.png", "Raw category labels", "Category standardisation is necessary before modelling."),
        ("fig07_outlier_flagged_retained.png", "Retained outliers", "Extreme records are flagged for review while valid observations stay in scope."),
        ("fig08_price_distribution_retained.png", "Retained price coverage", "Preparation preserves the breadth of Malaysia's housing market."),
    ],
    "Area quality and coverage": [
        ("fig04_tenure_before_after.png", "Tenure cleaning", "Equivalent mixed-tenure labels are consolidated."),
        ("fig05_type_before_after.png", "Property type cleaning", "Raw type text becomes a consistent Primary Type."),
        ("fig06_area_labels_before_after.png", "Area key preparation", "Clean Area and State are combined to prevent place-name ambiguity."),
        ("fig09_area_frequency_bands.png", "Area frequency", "Many areas have limited records, which raises uncertainty for rare locations."),
        ("fig10_state_counts.png", "State coverage", "Record coverage varies across states."),
    ],
    "Location and property": [
        ("fig13_state_price.png", "Price by state", "Location is a major source of price variation."),
        ("fig15_property_type_price.png", "Price by property type", "Housing formats occupy different price bands."),
        ("fig16_tenure_price.png", "Price by tenure", "Tenure adds a smaller but meaningful market distinction."),
        ("fig18_state_psf.png", "PSF by state", "Median PSF varies materially by geography."),
        ("fig19_psf_price_by_category.png", "PSF and price", "PSF is closely connected with predicted median price."),
        ("fig20_transactions_price.png", "Transactions and price", "Transaction count has a weaker relationship with price."),
    ],
}

FIGURE_TAKEAWAYS = {
    "Data quality": "The market is skewed and contains legitimate premium observations, so robust evaluation matters more than simply deleting extremes.",
    "Area quality and coverage": "Clean location keys protect model meaning, but rare and unseen areas still require transparent uncertainty warnings.",
    "Location and property": "Geography, property type, and especially independently sourced PSF drive most of the estimate's practical variation.",
}


def render_plotly(fig):
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=45, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
                      font=dict(color="#0F172A"), hoverlabel=dict(bgcolor="#FFFFFF"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def insights_page(data):
    st.markdown(
        f'<div class="mh-hero"><div><h1>Market Insights</h1><p>Filter the 2025 housing dataset, '
        'compare market segments, and interpret the evidence behind the estimator.</p></div>'
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
        if state != "All": subset = subset[subset["State"].eq(state)]
        if area != "All": subset = subset[subset["Area_Clean"].eq(area_labels[area])]
        if ptype != "All": subset = subset[subset["Primary_Type"].eq(ptype)]
        if tenure != "All": subset = subset[subset["Tenure"].eq(tenure)]
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
            left, right = st.columns(2)
            state_summary = subset.groupby("State", as_index=False)["Median_Price"].median().sort_values("Median_Price", ascending=False)
            with left:
                render_plotly(px.bar(state_summary, x="State", y="Median_Price", title="Median price by state",
                    color="Median_Price", color_continuous_scale=["#DBEAFE", "#2563EB"], labels={"Median_Price":"Median price (RM)"}))
            with right:
                render_plotly(px.box(subset, x="Primary_Type", y="Median_Price", color="Primary_Type",
                    title="Price distribution by property type", labels={"Median_Price":"Median price (RM)"}))
            left, right = st.columns(2)
            with left:
                render_plotly(px.scatter(subset, x="Median_PSF", y="Median_Price", color="Primary_Type",
                    size="Transactions", hover_data=["State", "Area_Clean"], title="Median PSF vs median price",
                    labels={"Median_PSF":"Median PSF (RM)", "Median_Price":"Median price (RM)"}))
            top_areas = (subset.groupby(["State", "Area_Clean"], as_index=False)["Median_Price"].median()
                         .nlargest(10, "Median_Price").sort_values("Median_Price"))
            top_areas["Area"] = top_areas["Area_Clean"].map(display_name) + ", " + top_areas["State"]
            with right:
                render_plotly(px.bar(top_areas, x="Median_Price", y="Area", orientation="h",
                    title="Top 10 areas by median price", color_discrete_sequence=["#0D9488"],
                    labels={"Median_Price":"Median price (RM)"}))
        else:
            st.info("Plotly is unavailable, so simplified Streamlit charts are shown.")
            state_summary = subset.groupby("State")["Median_Price"].median().sort_values(ascending=False)
            st.bar_chart(state_summary)
            st.scatter_chart(subset, x="Median_PSF", y="Median_Price", color="Primary_Type")
        export_columns = [c for c in ["Township", "Area_Clean", "State", "Primary_Type", "Tenure", "Median_Price", "Median_PSF", "Transactions"] if c in subset.columns]
        with st.expander("View filtered records"):
            st.dataframe(subset[export_columns], use_container_width=True, hide_index=True)
        st.download_button("Download filtered records (CSV)", data=subset[export_columns].to_csv(index=False).encode("utf-8"),
                           file_name="malaysia_housing_filtered.csv", mime="text/csv", key="download_filtered")
    else:
        category = st.selectbox("Insight category", list(FIGURE_GROUPS), key="insight_category")
        st.markdown(f'<div class="mh-takeaway"><strong>Key takeaway from this category</strong><br>{FIGURE_TAKEAWAYS[category]}</div>', unsafe_allow_html=True)
        for filename, title, insight in FIGURE_GROUPS[category]:
            st.markdown(
                f'<div class="mh-figure-card"><div class="mh-figure-title">{svg_icon("chart",18)} {escape(title)}</div>'
                f'<div class="mh-figure-insight">{escape(insight)}</div>'
                '<div class="mh-why"><strong>Why this matters:</strong> it clarifies the evidence, assumptions, or limitations used by the estimator.</div></div>',
                unsafe_allow_html=True,
            )
            figure_path = FIGURES_DIR / filename
            if figure_path.exists():
                st.image(str(figure_path), use_container_width=True)
            else:
                st.warning(f"Missing figure file: {filename}")

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
        "Performance": [("fig25_model_rmse_comparison.png", "Cross-validation and test RMSE"),
                        ("fig26_group_cv_fold_rmse.png", "Unseen-area fold stability"),
                        ("fig27_train_test_overfitting.png", "Train-test overfitting check")],
        "Diagnostics and dependency": [("fig28_median_psf_dependency.png", "Median PSF dependency"),
                                       ("fig29_prediction_diagnostics.png", "Prediction diagnostics")],
        "Importance": [("fig30_permutation_importance.png", "Permutation importance"),
                       ("fig31_native_importance.png", "Grouped native importance")],
    }
    section = st.selectbox("Diagnostic section", list(sections), key="model_report_section")
    for filename, title in sections[section]:
        figure_path = FIGURES_DIR / filename
        st.markdown(f'<div class="mh-figure-card"><div class="mh-figure-title">{svg_icon("model",18)} {escape(title)}</div></div>', unsafe_allow_html=True)
        if figure_path.exists():
            st.image(str(figure_path), use_container_width=True)
        else:
            st.warning(f"Missing figure file: {filename}")
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
