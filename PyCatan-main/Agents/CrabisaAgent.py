from Classes.Constants import *
from Classes.Board import Board
from Classes.DevelopmentCards import *
from Classes.Hand import Hand
from Classes.TradeOffer import TradeOffer
from Classes.Materials import Materials
import random
from Interfaces.AgentInterface import AgentInterface
from Managers.GameManager import GameManager

class CrabisaAgent(AgentInterface):
    """
    Interfaz que implementa a un agente
    """

    def __init__(self, agent_id):
        self.town_number = 0
        self.first_road_to = -1
        self.second_road_to = -1
        self.hand = Hand()
        self.board = Board()
        self.development_cards_hand = DevelopmentCardsHand()
        self.id = agent_id
        self.game_manager = GameManager(True)
        self.turn = 0
        self.material_given_more_than_three = None
        # Son los materiales más necesarios en construcciones, luego se piden con year of plenty para tener en mano
        self.year_of_plenty_material_one = MaterialConstants.CEREAL
        self.year_of_plenty_material_two = MaterialConstants.MINERAL

    # Los triggers son llamados por el GameDirector las veces que sean necesarias hasta que devuelvan null
    #  o el GameDirector le niegue continuar el trigger
    def on_trade_offer(self, board_instance, offer=TradeOffer(), player_making_offer=int):
        """
        Hay que tener en cuenta que gives se refiere a los materiales que da el jugador que hace la oferta,
        luego en este caso es lo que recibe
        :param offer:
        :return:
        """
        # TODO: ver los recursos que tenemos
        # TODO: evaluar las construcciones que podemos hacer
        recursos_propios = Hand()
        recursos_propios.add_material(MaterialConstants.CEREAL, self.hand.resources.cereal)
        recursos_propios.add_material(MaterialConstants.MINERAL, self.hand.resources.mineral)
        recursos_propios.add_material(MaterialConstants.CLAY, self.hand.resources.clay)
        recursos_propios.add_material(MaterialConstants.WOOD, self.hand.resources.wood)
        recursos_propios.add_material(MaterialConstants.WOOL, self.hand.resources.wool)

        recursos_que_nos_ofrece = Materials(0,0,0,0,0)
        recursos_que_nos_ofrece.add_from_id(MaterialConstants.CEREAL, offer.gives.cereal)
        recursos_que_nos_ofrece.add_from_id(MaterialConstants.MINERAL, offer.gives.mineral)
        recursos_que_nos_ofrece.add_from_id(MaterialConstants.CLAY,offer.gives.clay)
        recursos_que_nos_ofrece.add_from_id(MaterialConstants.WOOD, offer.gives.wood)
        recursos_que_nos_ofrece.add_from_id(MaterialConstants.WOOL, offer.gives.wool)

        recursos_que_nos_pide = Materials(0,0,0,0,0)
        recursos_que_nos_pide.add_from_id(MaterialConstants.CEREAL, offer.receives.cereal)
        recursos_que_nos_pide.add_from_id(MaterialConstants.MINERAL, offer.receives.mineral)
        recursos_que_nos_pide.add_from_id(MaterialConstants.CLAY,offer.receives.clay)
        recursos_que_nos_pide.add_from_id(MaterialConstants.WOOD, offer.receives.wood)
        recursos_que_nos_pide.add_from_id(MaterialConstants.WOOL, offer.receives.wool)


        construcciones_que_podemos_hacer = []
        
        if recursos_propios.resources.has_more(BuildConstants.CITY):
            construcciones_que_podemos_hacer.append(BuildConstants.CITY)
            recursos_propios.remove_material(MaterialConstants.CEREAL, 2)
            recursos_propios.remove_material(MaterialConstants.MINERAL, 3)
        if recursos_propios.resources.has_more(BuildConstants.TOWN):
            construcciones_que_podemos_hacer.append(BuildConstants.TOWN)
            recursos_propios.remove_material(MaterialConstants.CEREAL, 1)
            recursos_propios.remove_material(MaterialConstants.MINERAL, 1)
            recursos_propios.remove_material(MaterialConstants.CLAY, 1)
            recursos_propios.remove_material(MaterialConstants.WOOD, 1)
        if recursos_propios.resources.has_more(BuildConstants.ROAD):
            construcciones_que_podemos_hacer.append(BuildConstants.ROAD)
            recursos_propios.remove_material(MaterialConstants.CLAY, 1)
            recursos_propios.remove_material(MaterialConstants.WOOD, 1)
        if recursos_propios.resources.has_more(BuildConstants.CARD):
            construcciones_que_podemos_hacer.append(BuildConstants.CARD)
            recursos_propios.remove_material(MaterialConstants.CEREAL, 1)
            recursos_propios.remove_material(MaterialConstants.WOOL, 1)
            recursos_propios.remove_material(MaterialConstants.MINERAL, 1)

        recursos_propios_despues_de_trade = recursos_propios
        # en recursos propios tenemos ahora lo que nos sobra despues de hacer las construcciones
        # evaluamos si con lo que nos ofrece hacemos alguna construccion
        recursos_propios_despues_de_trade.add_material(MaterialConstants.CEREAL, recursos_que_nos_ofrece.cereal)
        recursos_propios_despues_de_trade.add_material(MaterialConstants.MINERAL, recursos_que_nos_ofrece.mineral)
        recursos_propios_despues_de_trade.add_material(MaterialConstants.CLAY, recursos_que_nos_ofrece.clay)
        recursos_propios_despues_de_trade.add_material(MaterialConstants.WOOD, recursos_que_nos_ofrece.wood)
        recursos_propios_despues_de_trade.add_material(MaterialConstants.WOOL, recursos_que_nos_ofrece.wool)
        recursos_propios_despues_de_trade.add_material(MaterialConstants.CEREAL, -recursos_que_nos_pide.cereal)
        recursos_propios_despues_de_trade.add_material(MaterialConstants.MINERAL, -recursos_que_nos_pide.mineral)
        recursos_propios_despues_de_trade.add_material(MaterialConstants.CLAY, -recursos_que_nos_pide.clay)
        recursos_propios_despues_de_trade.add_material(MaterialConstants.WOOD, -recursos_que_nos_pide.wood)
        recursos_propios_despues_de_trade.add_material(MaterialConstants.WOOL, -recursos_que_nos_pide.wool)
        # si despues del trade podemos hacer una construccion, aceptamos
        if recursos_propios_despues_de_trade.resources.has_more(BuildConstants.CITY):
            return True
        if recursos_propios_despues_de_trade.resources.has_more(BuildConstants.TOWN):
            return True
        if recursos_propios_despues_de_trade.resources.has_more(BuildConstants.ROAD):
            return True
        if recursos_propios_despues_de_trade.resources.has_more(BuildConstants.CARD):
            return True
        # si no podemos hacer ninguna construccion, 
        # contraofertamos manteniendo lo que nos ofrece (porque puede no tener otros materiales)
        # y dandole materiales que nos sobren hasta igualar la cantidad que nos pide
        # si no nos sobra nada, rechazamos
        cereal_a_dar = 0
        mineral_a_dar = 0
        clay_a_dar = 0
        wood_a_dar = 0
        wool_a_dar = 0
        cantidad_de_recursos_que_damos = 0
        cantidad_de_recursos_que_nos_ofrece = recursos_que_nos_ofrece.cereal + recursos_que_nos_ofrece.mineral + recursos_que_nos_ofrece.clay + recursos_que_nos_ofrece.wood + recursos_que_nos_ofrece.wool

        for i in range(cantidad_de_recursos_que_nos_ofrece):
            if recursos_propios.resources.wool > 2 and cantidad_de_recursos_que_damos < cantidad_de_recursos_que_nos_ofrece:
                wool_a_dar += 1
                cantidad_de_recursos_que_damos += 1
            if recursos_propios.resources.cereal > 2 and cantidad_de_recursos_que_damos < cantidad_de_recursos_que_nos_ofrece:
                cereal_a_dar += 1
                cantidad_de_recursos_que_damos += 1
            if recursos_propios.resources.mineral > 4 and cantidad_de_recursos_que_damos < cantidad_de_recursos_que_nos_ofrece:
                mineral_a_dar += 1
                cantidad_de_recursos_que_damos += 1
            if recursos_propios.resources.clay > 2 and cantidad_de_recursos_que_damos < cantidad_de_recursos_que_nos_ofrece:
                clay_a_dar += 1
                cantidad_de_recursos_que_damos += 1
            if recursos_propios.resources.wood > 2 and cantidad_de_recursos_que_damos < cantidad_de_recursos_que_nos_ofrece:
                wood_a_dar += 1
                cantidad_de_recursos_que_damos += 1
        if cantidad_de_recursos_que_damos == cantidad_de_recursos_que_nos_ofrece:
            return TradeOffer(Materials(cereal_a_dar, mineral_a_dar, clay_a_dar, wood_a_dar, wool_a_dar), recursos_que_nos_ofrece)  
        # Si no se ha devuelto nada, se rechaza la oferta
        return False
        


    def on_turn_start(self):
        # Si tiene mano de cartas de desarrollo
        if len(self.development_cards_hand.hand):
            # Mira todas las cartas
            for i in range(0, len(self.development_cards_hand.hand)):
                # Si una es un caballero
                if self.development_cards_hand.hand[i].type == DevelopmentCardConstants.KNIGHT:
                    # Si el ladrón está en nuestro terreno, la juega
                    for terrain in self.board.terrain:
                        if terrain['has_thief']:
                            for node_id in terrain['contacting_nodes']:
                                if self.board.nodes[node_id]['player'] == self.id:
                                    return self.development_cards_hand.select_card(i) 
        return None

    def on_having_more_than_7_materials_when_thief_is_called(self):
        # Comprueba si tiene materiales para construir una ciudad. Si los tiene, descarta el resto que no le sirvan.
        if self.hand.resources.has_more(BuildConstants.CITY):
            while self.hand.get_total() > 7:
                if self.hand.resources.wool > 0:
                    self.hand.remove_material(4, 1)

                if self.hand.resources.cereal > 2:
                    self.hand.remove_material(0, 1)
                if self.hand.resources.mineral > 3:
                    self.hand.remove_material(1, 1)

                if self.hand.resources.clay > 0:
                    self.hand.remove_material(2, 1)
                if self.hand.resources.wood > 0:
                    self.hand.remove_material(3, 1)
        # Si no tiene materiales para hacer una ciudad comprueba si tiene para hacer un pueblo
        elif self.hand.resources.has_more(BuildConstants.TOWN):
            while self.hand.get_total() > 7:
                if self.hand.resources.wool > 0:
                    self.hand.remove_material(4, 1)

                if self.hand.resources.cereal > 0:
                    self.hand.remove_material(0, 1)
                if self.hand.resources.mineral > 0:
                    self.hand.remove_material(1, 1)

                if self.hand.resources.clay > 0:
                    self.hand.remove_material(2, 1)
                if self.hand.resources.wood > 0:
                    self.hand.remove_material(3, 1)
        # Si no tiene para hacer un pueblo, comprueba si tiene para hacer una carta
        elif self.hand.resources.has_more(BuildConstants.CARD):
            while self.hand.get_total() > 7:
                if self.hand.resources.cereal > 0:
                    self.hand.remove_material(0, 1)
                if self.hand.resources.wool > 0:
                    self.hand.remove_material(4, 1)
                if self.hand.resources.mineral > 0:
                    self.hand.remove_material(1, 1)
                
                if self.hand.resources.clay > 0:
                    self.hand.remove_material(2, 1)
                if self.hand.resources.wood > 0:
                    self.hand.remove_material(3, 1)
        # Si no tiene para hacer una carta, descarta lo que menos le sirva
        return self.hand

    def on_moving_thief(self):
        #TODO: Bloquea un número 6 u 8 donde no tenga un pueblo, pero que tenga dos del rival
        #  sino Bloquea un número 6 u 8 donde no tenga un pueblo, pero que tenga uno del rival
        #  sino Bloquea un número 5 u 9 donde no tenga un pueblo, pero que tenga dos del rival
        #  sino Bloquea un número 5 u 9 donde no tenga un pueblo, pero que tenga uno del rival
        # Si no se dan las condiciones lo deja donde está, lo que hace que el GameManager lo ponga en un lugar aleatorio
        terrain_with_thief_id = -1
        for terrain in self.board.terrain:
            if not terrain['has_thief']:
                if terrain['probability'] == 6 or terrain['probability'] == 8:
                    nodes = self.board.__get_contacting_nodes__(terrain['id'])
                    has_own_town = False
                    has_enemy_town = False
                    enemy_town_quantity = 0
                    enemy = -1
                    for node_id in nodes:
                        if self.board.nodes[node_id]['player'] == self.id:
                            has_own_town = True
                            break
                        if self.board.nodes[node_id]['player'] != -1:
                            has_enemy_town = True
                            enemy_town_quantity += 1
                            enemy = self.board.nodes[node_id]['player']

                    if not has_own_town and has_enemy_town and enemy_town_quantity > 1:
                        return {'terrain': terrain['id'], 'player': enemy}
                    elif not has_own_town and has_enemy_town:
                        return {'terrain': terrain['id'], 'player': enemy}
                elif terrain['probability'] == 5 or terrain['probability'] == 9:
                    nodes = self.board.__get_contacting_nodes__(terrain['id'])
                    has_own_town = False
                    has_enemy_town = False
                    enemy_town_quantity = 0
                    enemy = -1
                    for node_id in nodes:
                        if self.board.nodes[node_id]['player'] == self.id:
                            has_own_town = True
                            break
                        if self.board.nodes[node_id]['player'] != -1:
                            has_enemy_town = True
                            enemy_town_quantity += 1
                            enemy = self.board.nodes[node_id]['player']

                    if not has_own_town and has_enemy_town and enemy_town_quantity > 1:
                        return {'terrain': terrain['id'], 'player': enemy}
                    elif not has_own_town and has_enemy_town:
                        return {'terrain': terrain['id'], 'player': enemy}
            else:
                terrain_with_thief_id = terrain['id']

        return {'terrain': terrain_with_thief_id, 'player': -1}

    def on_turn_end(self):
        # Si tiene mano de cartas de desarrollo
        if len(self.development_cards_hand.hand):
            # Mira todas las cartas
            for i in range(0, len(self.development_cards_hand.hand)):
                # Si una es un punto de victoria
                if self.development_cards_hand.hand[i].type == DevelopmentCardConstants.VICTORY_POINT:
                    # La juega
                    return self.development_cards_hand.select_card(i)
        return None

    def on_commerce_phase(self):
        """
        Se podría complicar mucho más la negociación, cambiando lo que hace en función de si tiene o no puertos y demás
        """
        # Juega monopolio si ha entregado más de 3 del mismo tipo de material a un jugador en el intercambio
        if self.material_given_more_than_three is not None:
            if len(self.development_cards_hand.hand):
                # Mira todas las cartas
                for i in range(0, len(self.development_cards_hand.hand)):
                    # Si una es un punto de monopolio
                    if self.development_cards_hand.hand[i].effect == DevelopmentCardConstants.MONOPOLY_EFFECT:
                        # La juega
                        return self.development_cards_hand.select_card(i)

        gives = Materials(0,0,0,0,0)
        receives = Materials(0,0,0,0,0)

        # No pide nada porque puede hacer una ciudad
        if self.town_number >= 1 and self.hand.resources.has_more(BuildConstants.CITY):
            self.material_given_more_than_three = None
            return None
        # Pedir lo que falte para una ciudad, ofrece el resto de materiales iguales a los que pide
        elif self.town_number >= 1:
            cereal_hand = self.hand.resources.cereal
            mineral_hand = self.hand.resources.mineral
            wood_hand = self.hand.resources.wood
            clay_hand = self.hand.resources.clay
            wool_hand = self.hand.resources.wool
            total_given_materials = (2 - cereal_hand) + (3 - mineral_hand)

            # Si hay más materiales que los pedidos
            if total_given_materials < (wood_hand + clay_hand + wool_hand):
                materials_to_give = [0, 0, 0, 0, 0]
                for i in range(0, total_given_materials):
                    # Se mezcla el orden de materiales
                    order = [MaterialConstants.CLAY, MaterialConstants.WOOD, MaterialConstants.WOOL]
                    random.shuffle(order)
                    # una vez mezclado se recorre el orden de los materiales y se coge el primero que tenga un valor
                    for mat in order:
                        if self.hand.resources.get_from_id(mat) > 0:
                            self.hand.remove_material(mat, 1)
                            materials_to_give[mat] += 1
                            break
                gives = Materials(materials_to_give[0], materials_to_give[1], materials_to_give[2],
                                  materials_to_give[3], materials_to_give[4])

            # Si no hay más materiales que los pedidos, simplemente se prueba a entregar todos lo que se tenga en mano
            else:
                gives = Materials(0, 0, clay_hand, wood_hand, wool_hand)

            receives = Materials(2, 3, 0, 0, 0)

        # Como no puede construir una ciudad pide materiales para hacer un pueblo
        elif self.town_number == 0:
            # Si tiene materiales para hacer un pueblo directamente no comercia
            if self.hand.resources.has_more(Materials(1, 0, 1, 1, 1)):
                return None
            # Si no los tiene hace un intercambio
            else:
                # Puedes cambiar materiales repetidos o minerales
                materials_to_receive = [0, 0, 0, 0, 0]
                materials_to_give = [0, 0, 0, 0, 0]

                number_of_materials_received = 0

                materials_to_receive[0] = 1 - self.hand.resources.cereal
                materials_to_receive[1] = 0 - self.hand.resources.mineral
                materials_to_receive[2] = 1 - self.hand.resources.clay
                materials_to_receive[3] = 1 - self.hand.resources.wood
                materials_to_receive[4] = 1 - self.hand.resources.wool

                # Nos aseguramos de que solo pida materiales que necesita, y que no hayan negativos
                for i in range(0, len(materials_to_receive)):
                    if materials_to_receive[i] <= 0:
                        materials_to_receive[i] = 0
                    else:
                        number_of_materials_received += 1

                # Por cada material que recibe, ofrece 1 de entre los que tiene en mano,
                #   pero guardándose al menos 1 de cada uno de los necesarios para hacer un pueblo
                for j in range(0, number_of_materials_received):
                    # Se mezcla el orden de materiales
                    order = [MaterialConstants.CEREAL, MaterialConstants.MINERAL, MaterialConstants.CLAY,
                             MaterialConstants.WOOD, MaterialConstants.WOOL]
                    random.shuffle(order)
                    # una vez mezclado se recorre el orden de los materiales y se coge el primero que tenga un valor
                    for mat in order:
                        if self.hand.resources.get_from_id(mat) > 1 or mat == MaterialConstants.MINERAL:
                            self.hand.remove_material(mat, 1)
                            materials_to_give[mat] += 1
                            break

                gives = Materials(materials_to_give[0], materials_to_give[1], materials_to_give[2],
                                  materials_to_give[3], materials_to_give[4])
                receives = Materials(materials_to_receive[0], materials_to_receive[1], materials_to_receive[2],
                                     materials_to_receive[3], materials_to_receive[4])

        trade_offer = TradeOffer(gives, receives)
        return trade_offer

    def on_build_phase(self, board_instance):
        # Juega año de la cosecha si le faltan 2 o 1 materiales para completar una construcción
        # Juega construir carreteras si le da para camino más largo o con ello puede alcanzar un puerto (que no tenga)
        self.board = board_instance

        # Si tiene mano de cartas de desarrollo
        if len(self.development_cards_hand.hand):
            # Mira todas las cartas
            for i in range(0, len(self.development_cards_hand.hand)):
                # Comprueba primero de que hay más de 2 carreteras disponibles para construirlas
                road_possibilities = self.board.valid_road_nodes(self.id)

                # Si una es año de la cosecha o construir carreteras y hay al menos 2 carreteras disponibles a construir
                if (self.development_cards_hand.hand[i].effect == DevelopmentCardConstants.YEAR_OF_PLENTY_EFFECT or
                        (self.development_cards_hand.hand[i].effect == DevelopmentCardConstants.ROAD_BUILDING_EFFECT and
                         len(road_possibilities) > 1)):
                    # La juega
                    return self.development_cards_hand.select_card(i)

        if self.hand.resources.has_more(BuildConstants.CITY) and self.town_number > 0:
            possibilities = self.board.valid_city_nodes(self.id)
            for node_id in possibilities:
                for terrain_piece_id in self.board.nodes[node_id]['contacting_terrain']:
                    # Hacemos una ciudad solo si la probabilidad de que salga el número es mayor o igual a 4/36
                    if self.board.terrain[terrain_piece_id]['probability'] == 5 or \
                            self.board.terrain[terrain_piece_id]['probability'] == 6 or \
                            self.board.terrain[terrain_piece_id]['probability'] == 8 or \
                            self.board.terrain[terrain_piece_id]['probability'] == 9:
                        self.town_number -= 1  # Transformamos un pueblo en una ciudad
                        return {'building': BuildConstants.CITY, 'node_id': node_id}

        if self.hand.resources.has_more(BuildConstants.TOWN):
            possibilities = self.board.valid_town_nodes(self.id)
            for node_id in possibilities:
                for terrain_piece_id in self.board.nodes[node_id]['contacting_terrain']:
                    # Hacemos un pueblo solo si la probabilidad de que salga el número es mayor o igual a 3/36
                    # O si el nodo es costero y posee un puerto
                    if self.board.terrain[terrain_piece_id]['probability'] == 4 or \
                            self.board.terrain[terrain_piece_id]['probability'] == 5 or \
                            self.board.terrain[terrain_piece_id]['probability'] == 6 or \
                            self.board.terrain[terrain_piece_id]['probability'] == 8 or \
                            self.board.terrain[terrain_piece_id]['probability'] == 9 or \
                            self.board.terrain[terrain_piece_id]['probability'] == 10:
                        self.town_number += 1  # Añadimos un pueblo creado
                        return {'building': BuildConstants.TOWN, 'node_id': node_id}

        if self.hand.resources.has_more(BuildConstants.ROAD):
            # Construye sí o sí carretera si acaba en un nodo costero, pero, ¿y si no lo busca aleatoriamente?
            # Idealmente, debe de poder buscar caminos y encontrar el ideal a un puerto o similar, pero eso implicaría
            #  programar un algoritmo de búsqueda de nodos por pesos que actualmente me parece imposible de hacer.

            # Comprobar qué caminos posibles hay para cada nodo. Escoger el más alto si el ID del nodo es 32 o más.
            # Más bajo si es menor hacer override de eso si uno de los dos es directamente costero

            # TODO: Sería ideal que funcionase pero hay poco tiempo, que coja una aleatoria, pero si es costero y tiene puerto lo coge siempre
            possibilities = self.board.valid_road_nodes(self.id)
            for road_obj in possibilities:
                if self.board.is_coastal_node(road_obj['finishing_node']) and \
                        self.board.nodes[road_obj['finishing_node']]['harbor'] != HarborConstants.NONE:
                    return {'building': BuildConstants.ROAD,
                            'node_id': road_obj['starting_node'],
                            'road_to': road_obj['finishing_node']}

            # Asumiendo que no hay ninguna ideal (es decir, robarse los puertos),
            #   construye una carretera aleatoria, el 60% de las veces
            will_build = random.randint(0, 2)
            if will_build:
                if len(possibilities):
                    road_node = random.randint(0, len(possibilities) - 1)
                    return {'building': BuildConstants.ROAD,
                            'node_id': possibilities[road_node]['starting_node'],
                            'road_to': possibilities[road_node]['finishing_node']}

        # Si tiene materiales para hacer una carta, la construye. Como va la última en la pila,
        #    ya habrá construido cualquier otra cosa más útil
        if self.hand.resources.has_more(BuildConstants.CARD):
            return {'building': BuildConstants.CARD}

        return None

    def on_game_start(self, board_instance):
        '''
        Para elegir el nodo donde construir la primera casa, se siguen los siguientes pasos:
        1. Se obtienen los nodos válidos donde se puede construir
        2. Calcula la probabilidad total de los terrenos adyacentes a cada nodo
        3. A esa probabilidad se le añadirá un factor extra 
            3.1 si tiene terrenos de madera o de ladrillo ya que son esenciales al principio
            3.2 si ademas tiene un puerto de 3:1
        4. Para la carretera se elige la que más nos beneficie
        '''
        self.board = board_instance
        possibilities = self.board.valid_starting_nodes()
        chosen_node_id = -1
        chosen_node_probability = -1
        chosen_road_to_id = -1
        probabilities = {
                        0:-1, 2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36,
                        7: 0,    8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36
                    }

        for node_id in possibilities:
            total_probability = 0
            extra_factor = 0
            for terrain_id in self.board.nodes[node_id]['contacting_terrain']:
                # Se añade la probabilidad de cada terreno adyacente en base a probabilities
                total_probability += probabilities[self.board.terrain[terrain_id]['probability']]
                # Se añade un factor extra si el terreno es de madera
                if self.board.terrain[terrain_id]['terrain_type'] == TerrainConstants.WOOD:
                    extra_factor += 0.1
            # si la probabilidad es mayor se elige ese nodo
            if total_probability > chosen_node_probability:
                chosen_node_id = node_id
                chosen_node_probability = total_probability

            #print("Nodo: ", node_id, "Probabilidad: ", total_probability)
            #print("Terrenos adyacentes: ", self.board.nodes[node_id]['contacting_terrain'])
            #for terrenoID in self.board.nodes[node_id]['contacting_terrain']:
                #terreno = self.board.terrain[terrenoID]
                #print("Terreno: ", terreno)
            
        #print("Nodo elegido: ", chosen_node_id, "probabilidad: ", total_probability)
        #for terrenoID in self.board.nodes[chosen_node_id]['contacting_terrain']:
            #terreno = self.board.terrain[terrenoID]
            #print("Terreno: ", terreno)
   
        # Se elige una carretera aleatoria entre todas las opciones
        possible_roads = self.board.nodes[chosen_node_id]['adjacent']
        chosen_road_to_id = possible_roads[random.randint(0, len(possible_roads) - 1)]
        return chosen_node_id, chosen_road_to_id
    
    

    def on_monopoly_card_use(self):
        # Elige el material que más haya intercambiado (variable global de esta clase)
        return self.material_given_more_than_three

    # noinspection DuplicatedCode
    def on_road_building_card_use(self):
        # Elige dos carreteras aleatorias entre las opciones
        valid_nodes = self.board.valid_road_nodes(self.id)
        # Se supone que solo se ha usado la carta si hay más de 2 carreteras disponibles a construir,
        # pero se dejan por si acaso
        if len(valid_nodes) > 1:
            while True:
                road_node = random.randint(0, len(valid_nodes) - 1)
                road_node_2 = random.randint(0, len(valid_nodes) - 1)
                if road_node != road_node_2:
                    return {'node_id': valid_nodes[road_node]['starting_node'],
                            'road_to': valid_nodes[road_node]['finishing_node'],
                            'node_id_2': valid_nodes[road_node_2]['starting_node'],
                            'road_to_2': valid_nodes[road_node_2]['finishing_node'],
                            }
        elif len(valid_nodes) == 1:
            return {'node_id': valid_nodes[0]['starting_node'],
                    'road_to': valid_nodes[0]['finishing_node'],
                    'node_id_2': None,
                    'road_to_2': None,
                    }
        return None

    def on_year_of_plenty_card_use(self):
        return {'material': self.year_of_plenty_material_one, 'material_2': self.year_of_plenty_material_two}
