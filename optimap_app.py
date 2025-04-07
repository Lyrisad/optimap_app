import math
import pandas as pd
import requests
from typing import List, Tuple, Set, Dict, Optional
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import random
from datetime import datetime, timedelta
import streamlit as st
import folium
from streamlit_folium import folium_static
import tempfile
import os
import json
import plotly.express as px
import plotly.graph_objects as go
import io
import sys

# ========================== Partie 1 : Logique d'optimisation ==========================

@dataclass
class Location:
    index: int
    lat: float
    lon: float
    address: str
    is_depot: bool = False
    service_time: float = 10.0  # Temps de service fixe de 10 minutes par défaut
    time_window: Optional[Tuple[float, float]] = None  # Fenêtre horaire optionnelle
    demand: float = 0.0  # Charge à ce point (optionnel)

@dataclass
class Route:
    locations: List[Location]
    total_time: float
    total_distance: float
    violations: Dict[str, float]  # Suivi des violations de contraintes
    total_demand: float = 0.0

class RouteOptimizer:
    def __init__(self, locations: List[Location], max_time: float, max_capacity: Optional[float] = None):
        self.locations = locations
        self.max_time = max_time
        self.max_capacity = max_capacity
        self.depot = locations[0]
        self.distance_matrix = self._calculate_distance_matrix()
        self.time_matrix = self._calculate_time_matrix()
        self.savings_matrix = self._calculate_savings_matrix()

    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calcule la matrice des distances entre toutes les adresses."""
        n = len(self.locations)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = haversine_distance(
                    self.locations[i].lat, self.locations[i].lon,
                    self.locations[j].lat, self.locations[j].lon
                )
                matrix[i][j] = matrix[j][i] = dist
        return matrix

    def _calculate_time_matrix(self) -> np.ndarray:
        """Calcule la matrice des temps entre toutes les adresses."""
        n = len(self.locations)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                time = travel_time(
                    self.locations[i].lat, self.locations[i].lon,
                    self.locations[j].lat, self.locations[j].lon
                )
                matrix[i][j] = matrix[j][i] = time
        return matrix

    def _calculate_savings_matrix(self) -> np.ndarray:
        """Calcule la matrice des économies avec l'algorithme de Clarke-Wright."""
        n = len(self.locations)
        savings = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                if i == 0 or j == 0:  # Ignorer le dépôt
                    continue
                savings[i][j] = (self.time_matrix[0][i] + self.time_matrix[0][j] - 
                               self.time_matrix[i][j])
                savings[j][i] = savings[i][j]
        return savings

    def _evaluate_route(self, route_indices: List[int]) -> Route:
        """Évalue une tournée et retourne ses métriques."""
        route_locations = [self.locations[i] for i in route_indices]
        total_time = 0
        total_distance = 0
        total_demand = 0
        current_time = 0
        violations = {
            'time': 0,
            'capacity': 0,
            'time_window': 0
        }

        for i in range(len(route_indices)-1):
            current = route_indices[i]
            next_loc = route_indices[i+1]
            
            # Temps de trajet
            travel_t = self.time_matrix[current][next_loc]
            total_time += travel_t
            current_time += travel_t
            
            # Distance parcourue
            total_distance += self.distance_matrix[current][next_loc]
            
            # Ajout de la charge (si capacité utilisée)
            if self.max_capacity is not None:
                total_demand += self.locations[next_loc].demand
            
            # Vérification de la fenêtre horaire (si définie)
            if self.locations[next_loc].time_window is not None:
                start_time, end_time = self.locations[next_loc].time_window
                if current_time > end_time:
                    violations['time_window'] += current_time - end_time
                elif current_time < start_time:
                    # Si on arrive trop tôt, attendre
                    current_time = start_time
            
            # Temps de service
            current_time += self.locations[next_loc].service_time

        # Vérification du dépassement du temps total
        if total_time > self.max_time:
            violations['time'] = total_time - self.max_time
        
        # Vérification du dépassement de capacité
        if self.max_capacity is not None and total_demand > self.max_capacity:
            violations['capacity'] = total_demand - self.max_capacity

        return Route(route_locations, total_time, total_distance, violations, total_demand)

    def _two_opt_swap(self, route: List[int], i: int, j: int) -> List[int]:
        """Applique un échange 2-opt sur la tournée."""
        new_route = route[:i]
        new_route.extend(reversed(route[i:j+1]))
        new_route.extend(route[j+1:])
        return new_route

    def _two_opt_optimize(self, route: List[int]) -> List[int]:
        """Optimise localement la tournée avec la méthode 2-opt."""
        best_route = route
        best_eval = self._evaluate_route(route)
        improved = True

        while improved:
            improved = False
            for i in range(1, len(route)-2):
                for j in range(i+1, len(route)-1):
                    new_route = self._two_opt_swap(best_route, i, j)
                    new_eval = self._evaluate_route(new_route)
                    
                    # Accepter si la distance est réduite et aucune violation n'est présente
                    if (new_eval.total_distance < best_eval.total_distance and 
                        all(v == 0 for v in new_eval.violations.values())):
                        best_route = new_route
                        best_eval = new_eval
                        improved = True
                        break
                if improved:
                    break
        return best_route

    def _find_best_insertion(self, route: List[int], unvisited: Set[int]) -> Tuple[int, int]:
        """Trouve la meilleure position pour insérer une adresse dans la tournée."""
        best_cost = float('inf')
        best_location = None
        best_position = None

        for loc in unvisited:
            for pos in range(1, len(route)):
                new_route = route[:pos] + [loc] + route[pos:]
                eval_route = self._evaluate_route(new_route)
                
                # Coût d'insertion basé sur la distance et les éventuelles violations
                cost = eval_route.total_distance
                if self.max_capacity is not None:
                    cost += sum(v * 1000 for v in eval_route.violations.values())
                
                if cost < best_cost:
                    best_cost = cost
                    best_location = loc
                    best_position = pos

        return best_location, best_position

    def optimize(self, num_vehicles: int) -> List[Route]:
        """Optimise les tournées en combinant l'algorithme de Clarke-Wright et une optimisation 2-opt."""
        routes = []
        unvisited = set(range(1, len(self.locations)))  # Exclure le dépôt

        while unvisited and len(routes) < num_vehicles:
            # Initialiser avec le dépôt
            route = [0]
            current_eval = self._evaluate_route(route)

            # Sélectionner le premier point optimal
            best_loc = None
            best_cost = float('inf')
            
            for loc in unvisited:
                new_route = [0, loc, 0]
                new_eval = self._evaluate_route(new_route)
                cost = new_eval.total_distance
                
                if cost < best_cost:
                    best_cost = cost
                    best_loc = loc
            
            if best_loc is None:
                break
                
            # Démarrer avec la meilleure tournée initiale
            route = [0, best_loc, 0]
            unvisited.remove(best_loc)
            current_eval = self._evaluate_route(route)

            while unvisited:
                best_loc, best_pos = self._find_best_insertion(route, unvisited)
                if best_loc is None:
                    break

                new_route = route[:best_pos] + [best_loc] + route[best_pos:]
                new_eval = self._evaluate_route(new_route)

                # Accepter l'insertion si aucune violation ou si les violations sont réduites
                if (all(v == 0 for v in new_eval.violations.values()) or 
                    sum(new_eval.violations.values()) < sum(current_eval.violations.values())):
                    route = new_route
                    unvisited.remove(best_loc)
                    current_eval = new_eval
                else:
                    break

            # Optimisation locale 2-opt
            route = self._two_opt_optimize(route)
            
            # Évaluation finale de la tournée
            final_eval = self._evaluate_route(route)
            routes.append(final_eval)

        return routes

def geocode_ban(full_address: str) -> Tuple[float, float]:
    """Géocode une adresse via l'API BAN."""
    url = "https://api-adresse.data.gouv.fr/search/"
    params = {'q': full_address, 'limit': 1}
    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        data = resp.json()
        if data.get('features'):
            coords = data['features'][0]['geometry']['coordinates']  # [lon, lat]
            lon, lat = coords[0], coords[1]
            return (lat, lon)
    raise ValueError(f"Impossible de localiser : {full_address}")

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcule la distance Haversine en km."""
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat/2)**2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(d_lon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def travel_time(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcule la durée (en minutes) pour aller de (lat1, lon1) à (lat2, lon2).
    Hypothèse : 1.2 min/km + 10 min de pause fixe.
    """
    dist_km = haversine_distance(lat1, lon1, lat2, lon2)
    drive_minutes = dist_km * 1.2
    pause = 10
    return drive_minutes + pause

def main():
    """
    Fonction principale pour exécuter l'optimisation en mode CLI.
    """
    # ---------------- Paramètres ----------------
    DAYS = 3
    VEHICLES_PER_DAY = 2
    HOURS_PER_DAY = 8
    MAX_TIME = HOURS_PER_DAY * 60  # 480 minutes
    MAX_CAPACITY = 100.0  # Capacité maximale par véhicule

    # ---------------- Lecture & géocodage ----------------
    df = pd.read_excel("Optimap.xlsx")
    locations = []

    # Création des objets Location
    for i, row in df.iterrows():
        ville = str(row["Villes"]).strip()
        adr = str(row["Adresses"]).strip()
        full_addr = f"{ville}, {adr}"
        
        try:
            lat, lon = geocode_ban(full_addr)
            # Génération d'exemples aléatoires pour les fenêtres horaires et les charges
            start_time = random.randint(0, 300)  # Début aléatoire entre 0 et 5h
            end_time = start_time + random.randint(60, 240)  # Fenêtre de 1 à 4h
            demand = random.uniform(0, 50)  # Charge aléatoire entre 0 et 50
            
            locations.append(Location(
                index=i,
                lat=lat,
                lon=lon,
                address=full_addr,
                is_depot=(i == 0),
                time_window=(start_time, end_time),
                demand=demand,
                service_time=random.uniform(5, 30)  # Temps de service entre 5 et 30 minutes
            ))
        except ValueError as e:
            print(f"Erreur de géocodage pour {full_addr}: {str(e)}")
            continue

    if len(locations) < 2:
        print("Il faut au moins 1 dépôt + 1 adresse.")
        return

    # ---------------- Optimisation des routes ---------------- 
    all_routes = []
    
    for day in range(1, DAYS + 1):
        unvisited_indices = [loc.index for loc in locations if not loc.is_depot and 
                           loc.index not in {r.locations[0].index for r in all_routes}]
        if not unvisited_indices:
            print(f"Toutes les adresses ont été visitées au jour {day}.")
            break

        # Constitution de la liste des adresses à optimiser (dépôt + non visités)
        locations_to_optimize = [locations[0]]
        locations_to_optimize.extend([locations[i] for i in unvisited_indices])
        
        optimizer = RouteOptimizer(locations_to_optimize, MAX_TIME, MAX_CAPACITY)
        day_routes = optimizer.optimize(VEHICLES_PER_DAY)
        
        all_routes.extend(day_routes)

    # ---------------- Affichage final ----------------
    print("\n=== Plan de route (Optimisation avancée) ===")
    for i, route in enumerate(all_routes):
        day = (i // VEHICLES_PER_DAY) + 1
        veh = (i % VEHICLES_PER_DAY) + 1
        
        print(f"\nJour {day}, Véhicule {veh}:")
        print(f"Temps total: {int(route.total_time)} min")
        print(f"Distance totale: {route.total_distance:.2f} km")
        print(f"Charge totale: {route.total_demand:.2f}")
        
        if any(route.violations.values()):
            print("⚠️ Violations:")
            for constraint, value in route.violations.items():
                if value > 0:
                    print(f"  - {constraint}: {value:.2f}")
        
        str_route = [loc.address for loc in route.locations]
        print(" -> ".join(str_route))

# ========================== Partie 2 : Interface Streamlit ==========================

def get_road_route(start_lat, start_lon, end_lat, end_lon):
    """Récupère l'itinéraire routier entre deux points via OSRM."""
    url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['code'] == 'Ok':
            return data['routes'][0]['geometry']['coordinates']
    return None

def create_route_map(locations, route_indices, title):
    """Crée une carte avec l'itinéraire routier."""
    m = folium.Map(
        location=[locations[0].lat, locations[0].lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    for i, loc in enumerate(locations):
        color = 'red' if i == 0 else 'blue'
        popup = f"{'DÉPÔT' if i == 0 else f'Point {i}'}: {loc.address}"
        folium.Marker(
            [loc.lat, loc.lon],
            popup=popup,
            icon=folium.Icon(color=color)
        ).add_to(m)
    
    for i in range(len(route_indices)-1):
        start_idx = route_indices[i]
        end_idx = route_indices[i+1]
        start_loc = locations[start_idx]
        end_loc = locations[end_idx]
        
        route_coords = get_road_route(
            start_loc.lat, start_loc.lon,
            end_loc.lat, end_loc.lon
        )
        
        if route_coords:
            # Conversion au format [lat, lon]
            route_coords = [[coord[1], coord[0]] for coord in route_coords]
            folium.PolyLine(
                route_coords,
                weight=3,
                color='red',
                opacity=0.8
            ).add_to(m)
    
    return m

def create_stats_plot(routes):
    """Crée des graphiques statistiques pour les tournées."""
    days = []
    vehicles = []
    distances = []
    times = []
    
    for i, route in enumerate(routes):
        day = (i // 2) + 1
        vehicle = (i % 2) + 1
        days.append(f"Jour {day}")
        vehicles.append(f"Véhicule {vehicle}")
        distances.append(route.total_distance)
        times.append(route.total_time)
    
    df = pd.DataFrame({
        'Jour': days,
        'Véhicule': vehicles,
        'Distance (km)': distances,
        'Temps (min)': times
    })
    
    fig1 = px.bar(df, x='Jour', y='Distance (km)', color='Véhicule',
                  title='Distance par jour et véhicule',
                  barmode='group')
    
    fig2 = px.bar(df, x='Jour', y='Temps (min)', color='Véhicule',
                  title='Temps par jour et véhicule',
                  barmode='group')
    
    return fig1, fig2

def adjust_column_widths(worksheet):
    """Ajuste la largeur des colonnes pour une meilleure lisibilité."""
    for column in worksheet.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = (max_length + 2) * 1.2
        worksheet.column_dimensions[column_letter].width = adjusted_width

def streamlit_main():
    st.set_page_config(page_title="OptiMap - Optimisation de Routes", layout="wide")

    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stAlert {
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0.5rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🚚 OptiMap - Optimisation de Routes")

    # Guide d'utilisation
    with st.expander("📖 Guide d'utilisation"):
        st.markdown("""
        ### Comment utiliser OptiMap ?
        1. **Configuration** : Ajustez les paramètres dans la barre latérale
            - Nombre de jours de tournée
            - Nombre de véhicules par jour
            - Heures de travail par jour
            - Temps de service par point
        2. **Import des données** :
            - Préparez un fichier Excel avec deux colonnes : "Villes" et "Adresses"
            - La première ligne doit être le dépôt
            - Chargez le fichier via le bouton "Choisir un fichier Excel"
        3. **Résultats** :
            - Visualisez les statistiques et les cartes
            - Explorez les détails de chaque tournée
            - Exportez les résultats dans différents formats
        """)

    # Paramètres dans la sidebar
    with st.sidebar:
        st.header("⚙️ Paramètres")
        days = st.number_input("Nombre de jours", min_value=1, value=3, help="Nombre de jours de tournée")
        vehicles_per_day = st.number_input("Véhicules par jour", min_value=1, value=2, help="Nombre de véhicules disponibles par jour")
        hours_per_day = st.number_input("Heures par jour", min_value=1, value=8, help="Durée de travail quotidienne en heures")
        service_time = st.slider("Temps de service par point (minutes)", min_value=20, max_value=30, value=25, help="Temps passé à chaque point de livraison")
        
        st.header("🔍 Filtres d'affichage")
        show_depot = st.checkbox("Afficher le dépôt", value=True, help="Inclure le dépôt dans les statistiques")
        show_stats = st.checkbox("Afficher les statistiques", value=True, help="Afficher les graphiques de performance")
        show_road_routes = st.checkbox("Afficher les routes routières", value=True, help="Afficher le tracé précis des routes sur la carte")

    # Chargement du fichier Excel
    uploaded_file = st.file_uploader("📂 Choisir un fichier Excel", type="xlsx", help="Format attendu : colonnes 'Villes' et 'Adresses'")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name

        try:
            df = pd.read_excel(temp_path)
            st.header("📍 Adresses à traiter")
            with st.expander("Voir les adresses"):
                st.dataframe(df)

            locations = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Géocodage des adresses
            for i, row in df.iterrows():
                ville = str(row["Villes"]).strip()
                adr = str(row["Adresses"]).strip()
                full_addr = f"{ville}, {adr}"
                
                try:
                    lat, lon = geocode_ban(full_addr)
                    locations.append(Location(
                        index=i,
                        lat=lat,
                        lon=lon,
                        address=full_addr,
                        is_depot=(i == 0),
                        service_time=service_time
                    ))
                    progress = (i + 1) / len(df)
                    progress_bar.progress(progress)
                    status_text.text(f"Géocodage: {i+1}/{len(df)} adresses traitées")
                except ValueError as e:
                    st.error(f"Erreur de géocodage pour {full_addr}: {str(e)}")
                    os.unlink(temp_path)
                    st.stop()

            optimizer = RouteOptimizer(locations, hours_per_day * 60)
            all_routes = optimizer.optimize(vehicles_per_day * days)

            # Vérification des adresses visitées et non visitées
            visited_locations = set()
            for route in all_routes:
                for loc in route.locations:
                    visited_locations.add(loc.address)
            
            all_locations = {loc.address for loc in locations if not loc.is_depot}
            unvisited_locations = all_locations - visited_locations
            
            if not unvisited_locations:
                st.success("✅ Toutes les adresses ont été visitées avec succès !")
            else:
                st.error("⚠️ Certaines adresses n'ont pas pu être visitées :")
                for addr in unvisited_locations:
                    st.write(f"- {addr}")
                st.write(f"Total: {len(unvisited_locations)} adresse(s) non visitée(s)")

            st.header("📊 Résumé Global")
            total_distance = sum(route.total_distance for route in all_routes)
            total_time = sum(route.total_time for route in all_routes)
            total_points = sum(len(route.locations) - 2 for route in all_routes)  # Exclure dépôt début/fin
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Distance totale", f"{total_distance:.1f} km")
            with col2:
                st.metric("Temps total", f"{int(total_time)} min")
            with col3:
                st.metric("Points visités", f"{total_points}")

            if show_stats:
                st.header("📈 Statistiques détaillées")
                tab1, tab2 = st.tabs(["📊 Graphiques", "📋 Tableau"])
                with tab1:
                    fig1, fig2 = create_stats_plot(all_routes)
                    st.plotly_chart(fig1, use_container_width=True)
                    st.plotly_chart(fig2, use_container_width=True)
                with tab2:
                    stats_data = []
                    for i, route in enumerate(all_routes):
                        day = (i // vehicles_per_day) + 1
                        veh = (i % vehicles_per_day) + 1
                        stats_data.append({
                            "Jour": f"Jour {day}",
                            "Véhicule": f"Véhicule {veh}",
                            "Points visités": len(route.locations) - 2,
                            "Distance (km)": f"{route.total_distance:.1f}",
                            "Temps (min)": int(route.total_time),
                            "Dépassement temps": f"{route.violations['time']:.1f}" if route.violations['time'] > 0 else "Non"
                        })
                    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

            st.header("🗺️ Plan des tournées")
            for i, route in enumerate(all_routes):
                day = (i // vehicles_per_day) + 1
                veh = (i % vehicles_per_day) + 1
                with st.expander(f"Jour {day}, Véhicule {veh}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Temps total", f"{int(route.total_time)} min")
                    with col2:
                        st.metric("Distance totale", f"{route.total_distance:.2f} km")
                    
                    if any(route.violations.values()):
                        st.warning("⚠️ Violations:")
                        for constraint, value in route.violations.items():
                            if value > 0 and constraint not in ['capacity', 'time_window']:
                                st.write(f"  - {constraint}: {value:.2f}")
                    
                    if show_road_routes:
                        route_map = create_route_map(
                            route.locations,
                            list(range(len(route.locations))),
                            f"Jour {day}, Véhicule {veh}"
                        )
                        folium_static(route_map)
                    
                    st.subheader("Détails de la tournée")
                    for j, loc in enumerate(route.locations):
                        if j == 0:
                            st.write(f"🚛 Départ: {loc.address}")
                        elif j == len(route.locations) - 1:
                            st.write(f"🏁 Retour: {loc.address}")
                        else:
                            st.write(f"📍 Point {j}: {loc.address}")
                            st.write(f"   ⏱️ Temps de service: {loc.service_time:.1f} min")

            st.header("📤 Export des résultats")
            export_format = st.selectbox("Format d'export", ["Excel détaillé", "Excel simplifié", "CSV", "JSON"])
            
            if "Excel" in export_format:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    summary_data = pd.DataFrame([{
                        "Total Distance (km)": f"{total_distance:.1f}",
                        "Total Temps (min)": int(total_time),
                        "Total Points": total_points,
                        "Nombre de véhicules": vehicles_per_day,
                        "Nombre de jours": days
                    }])
                    summary_data.to_excel(writer, sheet_name='Résumé', index=False)
                    adjust_column_widths(writer.sheets['Résumé'])
                    
                    for i, route in enumerate(all_routes):
                        day = (i // vehicles_per_day) + 1
                        veh = (i % vehicles_per_day) + 1
                        route_data = []
                        for j, loc in enumerate(route.locations):
                            if export_format == "Excel détaillé":
                                route_data.append({
                                    'Ordre': j,
                                    'Type': 'Dépôt' if j == 0 or j == len(route.locations)-1 else 'Point de livraison',
                                    'Adresse': loc.address,
                                    'Temps de service': loc.service_time,
                                    'Latitude': loc.lat,
                                    'Longitude': loc.lon
                                })
                            else:
                                route_data.append({
                                    'Ordre': j,
                                    'Type': 'Dépôt' if j == 0 or j == len(route.locations)-1 else 'Point de livraison',
                                    'Adresse': loc.address,
                                    'Temps de service': loc.service_time
                                })
                        df_route = pd.DataFrame(route_data)
                        sheet_name = f'Jour{day}_Veh{veh}'
                        df_route.to_excel(writer, sheet_name=sheet_name, index=False)
                        adjust_column_widths(writer.sheets[sheet_name])
                
                st.download_button(
                    label="📥 Télécharger Excel",
                    data=output.getvalue(),
                    file_name="routes_optimisees.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Télécharger les résultats au format Excel"
                )

        finally:
            os.unlink(temp_path)
    else:
        st.info("👆 Commencez par charger un fichier Excel contenant les adresses à traiter.")

# ========================== Point d'entrée ==========================

if __name__ == "__main__":
    # Pour exécuter en mode CLI, lancez avec l'argument "--cli"
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        main()
    else:
        streamlit_main()
