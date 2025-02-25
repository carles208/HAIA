import os
import random
import time
import imutils
import colorama
import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt

gpu = False
if gpu: import cupy as cp

POPULATION_SIZE = 500
ITERATIONS = 100

SHAPES_FOLDER_NAME = "Shapes"
SHAPES_FOLDER_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),SHAPES_FOLDER_NAME)

INITIAL_SHAPES_NAME = "init_Shapes"
INITIAL_SHAPES_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),INITIAL_SHAPES_NAME)

ORIGINAL_IMG_NAME = os.path.join("Original","original.png")
ORIGINAL_IMG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),ORIGINAL_IMG_NAME)

DOMINANT_COL_NUM = 20 # Número de colores dominantes usados en el inico de la generación

OBJECTS_NUM_IMG = 50 # Número de objetos en la imagen final

SURVIVOR_PER = 30 # Porcentaje de la población que sobrevive cada purga
SON_PER_SUR = int (POPULATION_SIZE / ((POPULATION_SIZE * (SURVIVOR_PER/100)) / 2))

MUTATE_PER = 0.7 # (1 - porcentaje de mutación de un hijo)
TRANSLATE_EVOL_MAX = 10 # Porcentaje máximo de pantalla que vamos a movernos
ROTATE_EVOL_MAX = 10 # Número de grados máximos que se va a rotar la imagen al mutar
SCALE_EVOL_MAX = 1.5 # Escalar máximo de tamaño a augmentar o disminuir 
COLOR_EVOL_MAX = 30 # Máximo valor rgb que puede mutar

COLOR_SP = 1
JOINED = 2
FIT_FUNC = COLOR_SP

SURV_TO_ADD = 1

OBJ_ADD = 1
ORIGINAL = cv.imread(ORIGINAL_IMG_DIR)
Ho, Wo, _ = np.shape(ORIGINAL)

class Genome:
    def __init__(self, ident:int, x:int, y:int, x_s:float , y_s:float , rot:float, r:float, g:float, b:float, img):
        self.ident   = ident
        self.x = x
        self.y = y
        self.x_s = x_s
        self.y_s = y_s
        self.rot = rot
        self.r = r
        self.g = g
        self.b = b
        self.img = img
    
    def __str__(self):
        return f"||GENOME Id: {self.ident}|| \n --x: {self.x} \n --y: {self.y} \n --x_s: {self.x_s} \n --y_s: {self.y_s} \n --rot: {self.rot} \n"
    
    def __repr__(self):
        # Este método define la representación que se usa al imprimir una lista de objetos
        return f"Genome({self.ident})"

def get_dominant_c(img, num):
    # Convertimos la imagen a espacio de color Lab
    img_lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)

    h, w, _ = np.shape(img_lab)

    # Preparamos los datos para K-means
    if img_lab.shape[2] == 4:
        data = np.reshape(img_lab[:, :, :3], (h * w, 3))  # Solo utiliza los primeros 3 canales
    else:
        data = np.reshape(img_lab, (h * w, 3))
    
    # Convertimos los datos a tipo float32 para K-means
    data = np.float32(data)

    # Parámetros del algoritmo k-means
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv.KMEANS_RANDOM_CENTERS

    # Aplicamos k-means
    compactness, labels, centers = cv.kmeans(data, num, None, criteria, 10, flags)

    # Convertimos los colores dominantes de nuevo a espacio BGR (opcional)
    centers_bgr = cv.cvtColor(np.uint8([centers]), cv.COLOR_Lab2BGR)[0]

    return centers_bgr  # Retorna los colores en BGR

def color_img(img, color):
    # Extraemos el alpha y el RGB de la imagen original
    bgr = img[:, :, :3].astype(np.float32)  # Convertir a float32
    alpha = img[:, :, 3].astype(np.float32)  # Convertir a float32

    # Creamos una máscara de color del mismo tamaño que la imagen (sin el canal alfa)
    color_mask = np.zeros_like(bgr)
    color_mask[:] = color

    # Comprobamos que las dimensiones de alpha y bgr coinciden
    if alpha.shape[:2] != bgr.shape[:2]:
        alpha = cv.resize(alpha, (bgr.shape[1], bgr.shape[0]), interpolation=cv.INTER_LINEAR)

    # Convertimos la imagen original al color usando la máscara
    alpha_scaled = alpha / 255.0  # Normalizar el canal alfa para usarlo en la fusión
    for c in range(3):
        bgr[:, :, c] = bgr[:, :, c] * (1 - alpha_scaled) + color_mask[:, :, c] * alpha_scaled
    

    # Convertir de nuevo a uint8 para guardarlo correctamente como imagen
    bgr = bgr.astype(np.uint8)

    # Combinamos los canales BGR coloreados con el canal alfa original
    imagen_coloreada = np.dstack((bgr, alpha.astype(np.uint8)))

    return imagen_coloreada

def rotate_img(img, degrees):
    # Reducir el tamaño de la imagen antes de rotar
    scale_factor = 0.5
    resized_img = cv.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_NEAREST)
    rotated_resized_img = imutils.rotate_bound(resized_img, degrees)
    # Volver a escalar la imagen al tamaño original
    original_size = (img.shape[1], img.shape[0])
    final_img = cv.resize(rotated_resized_img, original_size, interpolation=cv.INTER_NEAREST)
    return final_img

def scale_img(img, scale):
    return cv.resize(img, None, fx = scale[0], fy = scale[1], interpolation=cv.INTER_NEAREST)

def add_image(base_img, overlay_img, position):
    base_img = base_img.copy()
    x, y = position
    overlay_h, overlay_w = overlay_img.shape[:2]

    # Asegurarse de que las dimensiones no excedan las de la imagen base
    if y + overlay_h > base_img.shape[0]:
        overlay_h = base_img.shape[0] - y
    if x + overlay_w > base_img.shape[1]:
        overlay_w = base_img.shape[1] - x

    # Recortar overlay_img si es necesario
    overlay_img = overlay_img[:overlay_h, :overlay_w]

    # Recortar el área de la imagen base donde se colocará la overlay
    base_section = base_img[y:y+overlay_h, x:x+overlay_w]

    if overlay_img.shape[2] == 4:  # Si overlay tiene canal alpha
        # Separar los canales BGR y Alpha
        overlay_bgr = overlay_img[:, :, :3]
        overlay_alpha = overlay_img[:, :, 3] / 255.0  # Normalizar el canal alpha (0 a 1)

        # Crear una máscara inversa para la imagen base
        inv_alpha = 1.0 - overlay_alpha

        # Aplicar la transparencia (mezclar imágenes sin artefactos)
        for c in range(0, 3):
            base_section[:, :, c] = (overlay_alpha * overlay_bgr[:, :, c] +
                                     inv_alpha * base_section[:, :, c])
    else:
        # Si no tiene canal alpha, simplemente sustituir la sección correspondiente
        base_section[:, :, :3] = overlay_img[:, :, :3]

    # Asignar la sección modificada de nuevo a la imagen base
    base_img[y:y+overlay_h, x:x+overlay_w] = base_section

    return base_img

def generate_visible_position(original_img):
    """
    Genera una posición (x, y) dentro de las áreas visibles de la imagen original.
    Las áreas visibles se determinan por el canal alpha.
    """
    # Si la imagen tiene un canal alpha, trabajar con él
    if original_img.shape[2] == 4:  # Si hay canal alpha
        alpha_channel = original_img[:, :, 3]  # Extraer el canal alpha
        visible_positions = np.argwhere(alpha_channel > 0)  # Posiciones donde el alpha es mayor a 0
    else:
        # Si no hay canal alpha, cualquier posición es visible
        height, width = original_img.shape[:2]
        visible_positions = np.argwhere(np.ones((height, width), dtype=bool))

    selected_position = random.choice(visible_positions)

    return (selected_position[1],selected_position[0])

def init_population(size):
    # Cargamos las formas iniciales
    if os.path.isdir(SHAPES_FOLDER_DIR) == True:
        shapes = os.listdir(SHAPES_FOLDER_DIR)
    else:
        raise Exception("No se ha creado una carpeta de formas!")

    # Cargamos la imagen original
    original = cv.imread(ORIGINAL_IMG_DIR)
    h, w, _ = np.shape(original)
    centers = get_dominant_c(original, DOMINANT_COL_NUM)

    # Comprobamos la existencia de la carpeta de población inicial y si no existe creamos una
    # if (os.path.isdir(INITIAL_SHAPES_DIR) != True):
    #     print("--Initial shapes folder created--")
    #     os.mkdir(INITIAL_SHAPES_DIR)
    
    # Creamos el buffer de población inical
    población = []

    # Bucle de creación de genomas
    np_forma = (int) (POPULATION_SIZE / len(shapes)) # Número de genomas generados por formas base

    i = 0
    for shape in shapes:
        #print(os.path.join(SHAPES_FOLDER_DIR, shape))
        shape_i = cv.imread(os.path.join(SHAPES_FOLDER_DIR, shape),cv.IMREAD_UNCHANGED)
        for ident in range(i * np_forma , (i * np_forma) + np_forma):
            dominant_color = random.randint(0,len(centers)-1)
            rot = random.uniform(0,360)
            sx = random.uniform(0.1, w/200)
            sy = random.uniform(0.1, h/200)
            final_shape = color_img(shape_i, centers[dominant_color])
            final_shape = scale_img(final_shape, (sx, sy))
            final_shape = rotate_img(final_shape, rot) # Rotamos la imagen a la rotación rot
            x, y = generate_visible_position(original)
            g = Genome(ident, x, y, sx , sy, 
                       rot, centers[dominant_color][0],centers[dominant_color][1],
                       centers[dominant_color][2], final_shape) # Tocaria cambiar el 200 por el tamaño de la imagen de la forma o escalar la imagen

            # Guardamos las imágenes de la población inicial en la carpeta para visualizarlas
            cv.imwrite(os.path.join(INITIAL_SHAPES_DIR,(str)(g.ident) + ".png"), final_shape)

            # Añadimos el genoma a la población inicial
            población += [g]

        i += 1

    return población

def fitness(population, original_img, f_img):
    fitness_scores = []  # Lista para almacenar los scores de fitness
    # Imagen de referencia para agregar los genomas
    #f_img = img.copy()  # Imagen de referencia para agregar los genomas
    original_img_lab = cv.cvtColor(original_img, cv.COLOR_BGR2Lab)
        
    for genome in population:
        # Agregar la imagen del genoma a la imagen base
        a_img = add_image(f_img, genome.img, (genome.x, genome.y))
             
        # Convertir las imágenes a espacio de color Lab
        a_img_lab = cv.cvtColor(a_img, cv.COLOR_BGR2Lab)

        if gpu:
            # Ejemplo de conversión a CuPy si tus imágenes están en NumPy
            a_img_lab = cp.array(a_img_lab)  # Convertir a_img_lab de NumPy a CuPy
            original_img_lab = cp.array(original_img_lab)  # Convertir original_img_lab de NumPy a CuPy

            # Luego procede con el cálculo de la diferencia
            diff = cp.sqrt(cp.sum((a_img_lab - original_img_lab) ** 2, axis=-1))
            color_difference = cp.mean(diff)
        else:
            # Calcular la diferencia de color en el espacio Lab
            diff = np.sqrt(np.sum((a_img_lab - original_img_lab) ** 2, axis=-1))

            # Promedio de la diferencia de color
            color_difference = np.mean(diff)

        # Agregar el fitness score a la lista
        fitness_scores.append(color_difference)
        
    return fitness_scores

def delta_e_cie2000_vectorized(lab1, lab2):
    L1, a1, b1 = lab1[:, :, 0], lab1[:, :, 1], lab1[:, :, 2]
    L2, a2, b2 = lab2[:, :, 0], lab2[:, :, 1], lab2[:, :, 2]

    # Promedio de los valores L
    L_avg = (L1 + L2) / 2

    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2

    G = 0.5 * (1 - np.sqrt((C_avg**7) / (C_avg**7 + 25**7)))
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)

    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)
    C_avg_prime = (C1_prime + C2_prime) / 2

    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360
    delta_h_prime = h2_prime - h1_prime

    delta_h_prime = np.where(delta_h_prime > 180, delta_h_prime - 360, delta_h_prime)
    delta_h_prime = np.where(delta_h_prime < -180, delta_h_prime + 360, delta_h_prime)

    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    delta_H_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(delta_h_prime) / 2)

    L_avg_prime = (L1 + L2) / 2
    C_avg_prime = (C1_prime + C2_prime) / 2
    H_avg_prime = (h1_prime + h2_prime) / 2

    T = 1 - 0.17 * np.cos(np.radians(H_avg_prime - 30)) + 0.24 * np.cos(np.radians(2 * H_avg_prime)) + 0.32 * np.cos(np.radians(3 * H_avg_prime + 6)) - 0.20 * np.cos(np.radians(4 * H_avg_prime - 63))

    SL = 1 + (0.015 * (L_avg_prime - 50)**2) / np.sqrt(20 + (L_avg_prime - 50)**2)
    SC = 1 + 0.045 * C_avg_prime
    SH = 1 + 0.015 * C_avg_prime * T

    delta_theta = 30 * np.exp(-((H_avg_prime - 275) / 25)**2)
    RC = 2 * np.sqrt((C_avg_prime**7) / (C_avg_prime**7 + 25**7))
    RT = -np.sin(np.radians(2 * delta_theta)) * RC

    delta_E = np.sqrt(
        (delta_L_prime / SL)**2 +
        (delta_C_prime / SC)**2 +
        (delta_H_prime / SH)**2 +
        RT * (delta_C_prime / SC) * (delta_H_prime / SH)
    )

    return delta_E

def fitness2000(population, original_img, f_img):
    fitness_scores = []  # Lista para almacenar los scores de fitness
    original_img_lab = cv.cvtColor(original_img, cv.COLOR_BGR2Lab).astype(np.float32) / 255.0
    
    # Variables para la fórmula Delta E
    lab_height, lab_width, _ = original_img_lab.shape
    
    for genome in population:
        # Agregar la imagen del genoma a la imagen base
        a_img = add_image(f_img, genome.img, (genome.x, genome.y))
        a_img_lab = cv.cvtColor(a_img, cv.COLOR_BGR2Lab).astype(np.float32) / 255.0
        
        # Calcular la diferencia con Delta E 2000 en modo vectorizado
        delta_e = delta_e_cie2000_vectorized(original_img_lab, a_img_lab)

        # Promedio de la diferencia de color
        avg_color_difference = np.mean(delta_e)
        fitness_scores.append(avg_color_difference)
        
    return fitness_scores

def fitness_A(population, original_img, img):
    fitness_scores = []  # Lista para almacenar los scores de fitness
    f_img = img.copy()   # Imagen de referencia para agregar los genomas
    
    # Convertir la imagen original a espacio de color Lab una vez fuera del bucle
    original_img_lab = cv.cvtColor(original_img, cv.COLOR_BGR2Lab)

    for genome in population:
        # Agregar la imagen del genoma a la imagen base
        a_img = add_image(f_img, genome.img.copy(), (genome.x, genome.y))

        # Convertir a_img a espacio de color Lab
        a_img_lab = cv.cvtColor(a_img, cv.COLOR_BGR2Lab)

        # Calcular la diferencia entre las imágenes en espacio Lab
        l_diff = np.abs(a_img_lab[..., 0] - original_img_lab[..., 0])  # Canal L
        a_diff = np.abs(a_img_lab[..., 1] - original_img_lab[..., 1])  # Canal a
        b_diff = np.abs(a_img_lab[..., 2] - original_img_lab[..., 2])  # Canal b

        # Promediar las diferencias de color en Lab
        color_difference = (np.mean(l_diff) + np.mean(a_diff) + np.mean(b_diff)) / 3.0

        # Calcular la diferencia espacial (MSE) entre las imágenes
        spatial_difference = np.mean((a_img - original_img) ** 2)

        # Combinar ambas métricas
        combined_score = 0.1 * color_difference + 0.9 * spatial_difference

        # Agregar el fitness score a la lista
        fitness_scores.append(combined_score)
    
    return fitness_scores

def crossover(genome1, genome2):
    sons = []

    for i in range(SON_PER_SUR):
        son = Genome(
            ident=None,  # ID nuevo
            x=random.choice([genome1.x, genome2.x]),
            y=random.choice([genome1.y, genome2.y]),
            x_s=random.choice([genome1.x_s, genome2.x_s]),
            y_s=random.choice([genome1.y_s, genome2.y_s]),
            rot=random.choice([genome1.rot, genome2.rot]),
            r=random.choice([genome1.r, genome2.r]),
            g=random.choice([genome1.g, genome2.g]),
            b=random.choice([genome1.b, genome2.b]),
            img=random.choice([genome1.img.copy(), genome2.img.copy()])  # Puedes definir cómo combinar imágenes si es necesario
        )

        son = mutate(son) if random.random() >= MUTATE_PER else son

        sons.append(son)
    return sons

def mutate(gen):
    global Ho, Wo

    h, w, _ = np.shape(gen.img)

    new_x = max(0, min(gen.x + random.randint(0, TRANSLATE_EVOL_MAX) * random.choice([1, -1]), Wo - w))
    new_y = max(0, min(gen.y + random.randint(0, TRANSLATE_EVOL_MAX) * random.choice([1, -1]), Ho - h))

    new_sx = max(0.1, min(gen.x_s + random.uniform(0, SCALE_EVOL_MAX) * random.choice([1, -1]), 1.5))
    new_sy = max(0.1, min(gen.y_s + random.uniform(0, SCALE_EVOL_MAX) * random.choice([1, -1]), 1.5))

    rotation = random.uniform(0, ROTATE_EVOL_MAX) * random.choice([1, -1])
    new_rot = gen.rot + rotation

    new_r = min(255, max(0, gen.r + random.uniform(-COLOR_EVOL_MAX, COLOR_EVOL_MAX)))
    new_g = min(255, max(0, gen.g + random.uniform(-COLOR_EVOL_MAX, COLOR_EVOL_MAX)))
    new_b = min(255, max(0, gen.b + random.uniform(-COLOR_EVOL_MAX, COLOR_EVOL_MAX)))
    
    new_sx = new_sx if new_sx > 0 and new_sx*w >= 2 and new_sx * w < Wo*1.3 else 1
    new_sy = new_sy if new_sy > 0 and new_sy*h >= 2 and new_sx * h < Ho*1.3 else 1

    new_img = scale_img(gen.img.copy(), (new_sx, new_sy)) #if random.random() >= MUTATE_PER else gen.img.copy()
    new_img = rotate_img(new_img, rotation) #if random.random() >= MUTATE_PER else new_img # Rotamos la imagen a la rotación rot
    new_img = color_img(new_img, [new_r, new_g, new_b]) #if random.random() >= MUTATE_PER else new_img

    mutate = Genome(2,new_x, new_y, new_sx, new_sy, new_rot, new_r, new_g, new_b, new_img)

    return mutate

def select_survivors(population, fit):
    united = sorted(zip(population, fit),key= lambda x:x[1])
    del united[(int)(POPULATION_SIZE*0.1):]
    survivors_zip = united # Descartamos la mitad inferior del array 
    survivors, fit = zip(*survivors_zip) if survivors_zip else ([], [])
    return survivors

def progress_bar(progress, total, color=colorama. Fore. YELLOW):
    percent = 100 * (progress / float(total))
    bar = '█' * int(percent) + '-' * (100 - int(percent))
    print(color + f"\r{bar}| {percent: .2f}%", end="\r")
    if progress == total:
        print(colorama.Fore.GREEN + f"\r{bar}| {percent:.2f}%", end="\r")
        print()

def render():
    ruta_imagenes = "./Renders"
    imagenes = [img for img in os.listdir(ruta_imagenes) if img.endswith(".png")]
    imagenes.sort()  # Ordenar las imágenes por nombre

    # Leer la primera imagen para obtener las dimensiones
    imagen_inicial = cv.imread(os.path.join(ruta_imagenes, imagenes[0]))
    alto, ancho, _ = imagen_inicial.shape

    # Definir el codec y crear el objeto VideoWriter
    video_salida = cv.VideoWriter('video_salida.mp4', cv.VideoWriter_fourcc(*'mp4v'), 15, (ancho, alto))

    # Recorrer las imágenes y escribirlas en el video
    for imagen in imagenes:
        img = cv.imread(os.path.join(ruta_imagenes, imagen))
        video_salida.write(img)

    # Liberar el objeto VideoWriter
    video_salida.release()

def genetic_algorithm():
    print(colorama.Fore.RESET)
    global TRANSLATE_EVOL_MAX, POPULATION_SIZE, ITERATIONS

    print("¡Inicio del algoritmo genetico!")


    original_img = cv.imread(ORIGINAL_IMG_DIR)

    h, w, _ = np.shape(original_img)
    TRANSLATE_EVOL_MAX = (int) ((TRANSLATE_EVOL_MAX / 100) * min(h, w))

    # Si la imagen original no tiene alpha se lo añadimos
    if original_img.shape[2] != 4:
        alpha = np.zeros((original_img.shape[0], original_img.shape[1], 4), dtype=np.uint8)
        alpha[:, :, 3] = 255
        alpha[:, :, :3] = original_img
        original_img = alpha
    
    # Generamos la población inicial
    print(get_dominant_c(original_img,1))
    
    dominant_color = get_dominant_c(original_img, 4)[0]  # Obtener el primer color dominante

    # Crear una imagen llena con el color dominante
    new_img = np.zeros_like(original_img)  # Crear una imagen vacía del mismo tamaño
    new_img[:, :, 0] = dominant_color[0]  # Canal azul
    new_img[:, :, 1] = dominant_color[1]  # Canal verde
    new_img[:, :, 2] = dominant_color[2]  # Canal rojo
    new_img[:, :, 3] = 255  # Canal alfa completamente opaco
    
    last_fit = 100000000000000000000000000000000000000
    init = time.time()
    fitness_list = []
    for o in range(1, OBJECTS_NUM_IMG + 1):
        survivors = []
        population = []
        population = init_population(POPULATION_SIZE)
        print(f"Búsqueda genética objeto {o}:")
        # Algotimo genético
        start = time.time()
        for iter in range(1, ITERATIONS + 1):
            fit = []
            aux = new_img.copy() 
            fit = fitness2000(population, original_img, aux) if FIT_FUNC == COLOR_SP else fitness_A(population, original_img, aux)
            
            survivors = select_survivors(population, fit) # Futuro --> Añadir un parametro con el tipo de Método de criba
        
            population = list(survivors)
            for i in range(len(survivors) // 2):
                parent1 = survivors[i]
                parent2 = survivors[len(survivors) - 1 - i]
                sons = crossover(parent1, parent2)

                population += sons

            progress_bar(iter, ITERATIONS)
        
        if fit[0] < last_fit:
            new_img = add_image(new_img, survivors[0].img, [survivors[0].x, survivors[0].y]) # Falta hacer la update
            last_fit = fit[0]
            t_ob = time.time()-start
        print(f"Fitness: {last_fit*10}\nTiempo objeto:{t_ob}\nTiempo restante: {(t_ob * (OBJECTS_NUM_IMG - o))/60} minutos")
        
        fitness_list.append(last_fit)
        cv.imwrite("./Renders/" + (str)(o) + ".png",new_img)
        cv.imwrite("a.png",new_img)

    # Crear tablas de fitness y temperatura
    fitness_df = pd.DataFrame({
        "Iteration": range(1, len(fitness_list) + 1),
        "Fitness": fitness_list
    })

    print("¡Fin del algoritmo genetico!")

    plot_line_graph(fitness_df, "Fitness Progress", "Iteration", "Fitness", "fitness_progress.png")

    render()

def plot_line_graph(df, title, x_label, y_label, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_label], df[y_label], marker='o')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.savefig(filename, bbox_inches='tight', dpi=300)  # Guardar el gráfico como imagen PNG
    plt.show()


if __name__ == "__main__":
    #render()
    genetic_algorithm()

#Color dominante de las imágenes negras