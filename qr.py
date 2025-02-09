import cv2
import numpy as np
import reedsolo
import matplotlib.pyplot as plt


def masca0(M,i,j):
    if (i*j)%2+(i*j)%3 == 0:
        if M[i][j] == 1:
            M[i][j] = 0
        else:
            M[i][j] = 1
def masca1(M,i,j):
    if (i//2 + j // 3)%2 == 0:
        if M[i][j] == 1:
            M[i][j] = 0
        else:
            M[row][col] = 1
def masca2(M,i,j):
    if ((i * j) % 3 + (i +j)) % 2 == 0:
        if M[i][j] == 1:
            M[i][j] = 0
        else:
            M[i][j] = 1
def masca3(M,i,j):
    if ((i*j)%3 + (i*j))%2 == 0:
        if M[i][j] == 1:
            M[i][j] = 0
        else:
            M[i][j] = 1
def masca4(M,i,j):
    if  i % 2 == 0:
        if M[i][j] == 1:
            M[i][j] = 0
        else:
            M[i][j] = 1
def masca5(M,i,j):
    if (i + j) % 2 == 0:
        if M[i][j] == 1:
            M[i][j] = 0
        else:
            M[i][j] = 1
def masca6(M,i,j):
    if (i + j) % 3 == 0:
        if M[i][j] == 1:
            M[i][j] = 0
        else:
            M[i][j] = 1
def masca7(M,row,col):
    if col%3 == 0:
        if M[row][col] == 1:
            M[row][col] = 0
        else:
            M[row][col] = 1

def aplica_masca(matrice, tip_masca,excluded_coords):
    dim = len(matrice)
    for i in range(dim):
        for j in range(dim):
            # Aplicăm funcția specifică fiecărei măști
            if tip_masca == 0:
                condition=(i + j) % 2 == 0
            elif tip_masca == 1:
                condition = i % 2 == 0
            elif tip_masca == 2:
                condition =  j % 3 == 0
            elif tip_masca == 3:
                condition = (i + j) % 3 == 0
            elif tip_masca == 4:
                condition = (i // 2 + j // 3) % 2 == 0
            elif tip_masca == 5:
                condition = (i * j) % 2 + (i * j) % 3 == 0
            elif tip_masca == 6:
                condition = ((i * j) % 2 + (i * j) % 3) % 2 == 0
            elif tip_masca == 7:
                condition = ((i + j) % 2 + (i * j) % 3) % 2 == 0
            if condition and (i,j) not in excluded_coords:
                if matrice[i][j] == 1:
                    matrice[i][j] = 0
                else :
                    matrice[i][j] = 1
    return matrice

def salveaza_qr_din_matrice(matrice, nume_fisier="qr_code.png"):
    """
    Generează și salvează codul QR în format imagine (PNG).
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(matrice, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.savefig(nume_fisier, bbox_inches='tight', pad_inches=0)
    plt.close()  # Închidem figura pentru a elibera memoria

def calculeaza_penalty(matrice):
    dim = len(matrice)
    penalty = 0

    # Regula 1: Secvențe de cel puțin 5 celule consecutive
    for i in range(dim):
        pen_row = eval_secvente(matrice[i])
        pen_col = eval_secvente([matrice[j][i] for j in range(dim)])
        penalty += pen_row + pen_col

    # Regula 2: Blocuri de celule 2x2
    for i in range(dim - 1):
        for j in range(dim - 1):
            if matrice[i][j] == matrice[i + 1][j] == matrice[i][j + 1] == matrice[i + 1][j + 1]:
                penalty += 3
    # Regula 3: Detectare pattern Finder
    penalty += detecteaza_finder_pattern(matrice)

    # Regula 4: Raport module 1 și module 0
    total_celule = dim * dim
    module_pline = sum([sum(row) for row in matrice])
    procent_plin = (module_pline / total_celule) * 100
    diferenta = abs(procent_plin - 50)
    penalty += int(diferenta / 5) * 10  # Penalizare pentru fiecare 5% în exces

    return penalty

def eval_secvente(linie):

    lungime = len(linie)
    penalizare = 0
    count = 1
    for i in range(1, lungime):
        if linie[i] == linie[i - 1]:
            count += 1
            if count >= 5:
                penalizare += 1 if count == 5 else 1
        else:
            count = 1  # Resetez contorul dacă bitul diferă
    return penalizare

def detecteaza_finder_pattern(matrice):
    dim = len(matrice)
    penalty = 0
    pattern = [1, 0, 1, 1, 1, 0, 1]
    # Caut în rânduri
    for row in matrice:
        row_str = ''.join(map(str, row))
        penalty += row_str.count('1011101')
    # Caut în coloane
    for col in range(dim):
        col_values = ''.join(str(matrice[i][col]) for i in range(dim))
        penalty += col_values.count('1011101')

    return penalty * 40  # Penalizare este 40 per pattern
def calculeaza_format(ec_level, masca):
    cod_format_initial = (ec_level << 3) | masca  # Concatenăm ECL și masca (5 biți)
    generator_bch = 0b10100110111  # Polinom pentru checksum BCH
    cod_cu_spatiu = cod_format_initial << 10
    while cod_cu_spatiu.bit_length() > 10:
        shift = cod_cu_spatiu.bit_length() - generator_bch.bit_length()
        cod_cu_spatiu ^= generator_bch << shift
    checksum = cod_cu_spatiu
    cod_format_complet = (cod_format_initial << 10) | checksum
    # Aplicăm masca XOR finală pentru cod format
    masca_format = 0b101010000010010
    cod_format_final = cod_format_complet ^ masca_format
    return f"{cod_format_final:015b}"

def adauga_format_in_matrice(matrice, cod_format):
    dim = len(matrice)
    # Stânga-sus (biții pe linia 8, până la poziția 6)
    for i in range(6):
        matrice[8][i] = int(cod_format[i])
        matrice[i][8] = int(cod_format[i])
    matrice[8][7] = int(cod_format[6])
    matrice[7][8] = int(cod_format[6])

    for i in range(7):
        matrice[dim - 1 - i][8] = int(cod_format[7 + i])
        matrice[8][dim - 1 - i] = int(cod_format[7 +i])

caz = int(input("Ce vrei sa faci? Introdu cifra corespunzatoare:\n 1)Creeaza un cod qr dupa un sir de caractere \n 2)Interpreteaza un cod qr\n "))
if caz == 1:
    #lungimi maxime in functie de versiune
    versiuni = {1:17,2:32,3:53,4:78,5:106}
    nsymr = {1:7,2:10,3:15,4:20,5:26}
    sir = input("Introdu textul pe care vrei sa-l transformi intr-un cod qr:(maxim 106 caractere)\n")
    if len(sir)>108:
        print("Text prea lung")
    data = [0,1,0,0] #byte mode
    lungime = len(sir)
    #determin versiune
    if lungime>78:
        versiune = 5
    elif lungime>53:
        versiune = 4
    elif lungime>32:
        versiune = 3
    elif lungime>17:
        versiune = 2
    else:
        versiune = 1
    h=w=21 + (versiune-1)*4
    lungime = format(lungime,'08b')
    for bit in lungime:
        data.append(int(bit))
    litere_in_biti = [format(ord(litera),'08b') for litera in sir]
    for litera in litere_in_biti:
        for bit in litera:
            data.append(int(bit))
    if versiuni[versiune]*8-len(data)>0:
        if versiuni[versiune]*8-len(data)>=4:
            for i in range(4):
                data.append(0)
            while len(data)%8 !=0:
                data.append(0)
        else:
            for i in range(versiuni[versiune]*8-len(data)):
                data.append(0)
    padding="1110110000010001"
    x=0

    while versiuni[versiune]*8-len(data)>0:
        data.append(int(padding[x % len(padding)]))
        x+=1

    def aplica_reed_solomon(codewords,nsym):
        rs = reedsolo.RSCodec(nsym)
        try:
            encode = rs.encode(bytearray(codewords))
            return list(encode)
        except reedsolo.ReedSolomonError as e:
            print(f"Eroare Reed Solomon: {e}")
            return None

    data_str = ''.join(str(x) for x in data)
    codewords = [int(data_str[i:i+8],2) for i in range(0,len(data),8)]
    num_rs = nsymr[versiune]
    codewords_cu_rs = aplica_reed_solomon(codewords,num_rs)
    codewords_str = []
    for codeword in codewords_cu_rs:
        bits = format(codeword,'08b')
        codewords_str.extend(b for b in bits)
    print(codewords_cu_rs)
    for codeword in codewords_str:
        for bit in codeword:
            data.append(int(bit))

    M = np.zeros((h,w), dtype=int)
    for i in range(7):
        M[i][6] = 1
        M[6][i] = 1
        M[w-i-1][6]=1
        M[6][w-i-1]=1
        M[i][w-7]=1
        M[w-7][i]=1
        M[0][i]=1
        M[i][0]=1

        M[i][h-1]=1
        M[0][w-i-1]=1
        M[w-i-1][0]=1
        M[h-1][i] = 1

    for i in range(2,5):
        for j in range(2,5):
            M[i][j]=1

    for i in range(h-5,h-2):
        for j in range(2,5):
            M[i][j]=1

    for j in range(h-5,h-2):
        for i in range(2,5):
            M[i][j]=1

    for i in range(8,w-7):
        if i%2==0:
            M[6][i]=1
            M[i][6]=1
    alig = {2: (18, 18), 3: (22, 22), 4: (26, 26), 5: (30, 30), 6: (34, 34)}
    if versiune>=2:
        centru = alig[versiune]
        ax, ay = centru
        M[ax][ay] = 1
        for i in range(ax-2,ax+3):
            M[i][ay-2]=1
            M[i][ay+2]=1
        for j in range(ay-2,ay+3):
            M[ax-2][j]=1
            M[ax+2][j]=1
    M[h-8][8]=1

    # generez lista de elemente ce nu trebuie sa fie afectate de masca
    excluded_coords = []
    final = 21 + 4 * (versiune - 1)
    # Pozitionare+separator:
    pozitii = [(0, 0), (final - 8, 0), (0, final - 8)]
    for poz in pozitii:
        x, y = poz
        for row in range(x, x + 8):
            for col in range(y, y + 8):
                excluded_coords.append((row, col))
    # sincronizare:
    for i in range(final):
        excluded_coords.append((6, i))  # orizontală
        excluded_coords.append((i, 6))  # verticală
    # format
    for i in range(9):
        excluded_coords.append((8, i))
        excluded_coords.append((i, 8))
        if final - i < 29:
            excluded_coords.append((8, final - i))
            excluded_coords.append((final - i, 8))
    # aliniere, dictionar de valori predefinite centru
    if versiune > 1:
        centru = alig[versiune]
        ax, ay = centru
        for row in range(ax - 2, ax + 3):
            for col in range(ay - 2, ay + 3):
                excluded_coords.append((row, col))
    i = 0
    rows, cols = len(M), len(M[0])

    col = cols - 1  # Începem din colțul din dreapta jos
    row = rows - 1
    direction = -1  # -1 pentru sus, 1 pentru jos

    while col >= 0 and i<len(data):
        if col == 6:  # Coloana 6 este rezervată pentru pattern de sincronizare
            col -= 1

        while 0 <= row < rows and i<len(data):
            if (row, col) not in excluded_coords:  # Ignoră zonele fixe
                    M[row][col] = data[i]
                    i+=1
            if (row, col - 1) not in excluded_coords:
                if i < len(data):
                    M[row][col-1] = data[i]
                    i += 1

            row += direction  # Ne deplasăm sus sau jos

        direction *= -1  # Schimbăm direcția
        col -= 2  # Ne mutăm la următoarele două coloane din stânga
        row += direction


    masca_ideala = None
    scor_minim = float('inf')

    for tip_masca in range(8):
        masked_M = aplica_masca(M.copy(),tip_masca,excluded_coords)
        scor = calculeaza_penalty(masked_M)
        if scor < scor_minim:
            scor_minim = scor
            masca_ideala = tip_masca
    print(masca_ideala)
    masca = masca_ideala
    final = 21 + (versiune - 1) * 4
    M = aplica_masca(M,masca,excluded_coords)
    # if masca == 0:
    #     for row in range(final):
    #         for col in range(final):
    #             if (row, col) not in excluded_coords:
    #                 masca0(M, row, col)
    # elif masca == 1:
    #     for row in range(final):
    #         for col in range(final):
    #             if (row, col) not in excluded_coords:
    #                 masca1(M, row, col)
    # elif masca == 2:
    #     for row in range(final):
    #         for col in range(final):
    #             if (row, col) not in excluded_coords:
    #                 masca2(M, row, col)
    # elif masca == 3:
    #     for row in range(final):
    #         for col in range(final):
    #             if (row, col) not in excluded_coords:
    #                 masca3(M, row, col)
    # elif masca == 4:
    #     for row in range(final):
    #         for col in range(final):
    #             if (row, col) not in excluded_coords:
    #                 masca4(M, row, col)
    # elif masca == 5:
    #     for row in range(final):
    #         for col in range(final):
    #             if (row, col) not in excluded_coords:
    #                 masca5(M, row, col)
    # elif masca == 6:
    #     for row in range(final):
    #         for col in range(final):
    #             if (row, col) not in excluded_coords:
    #                 masca6(M, row, col)
    # elif masca == 7:
    #     for row in range(final):
    #         for col in range(final):
    #             if (row, col) not in excluded_coords:
    #                 masca7(M, row, col)
    print(M)
    masca =int(format(masca,'03b'),2)
    cod_format = calculeaza_format(0b01,masca)
    print(cod_format)
    adauga_format_in_matrice(M,cod_format)
    print(M)
    quiet_zone = 4
    M = np.pad(M, pad_width=quiet_zone, mode='constant', constant_values=0)
    plt.figure(figsize=(6, 6))

    # Afișăm matricea sub formă de imagine binară
    plt.imshow(M, cmap='binary', interpolation='nearest')
    plt.axis('off')  # Eliminăm axele pentru estetică

    # Afișăm codul QR generat
    plt.show()

elif caz == 2:
    #Prelucrare cod qr
    # Încarca qr-ul in format alb negru
    image_path = input("nume poza(qr_link.png pentru link si qr_nume.png pentru numele echipei: ")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Separ fundalul de qr,binarizez
    _, binarized_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)  # Inversăm (alb -> negru)
    # Găsim contururile din imagine
    contururi, _ = cv2.findContours(binarized_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contur_qr = max(contururi, key=cv2.contourArea)
    # Determinăm marginea si decupam codul
    x, y, w, h = cv2.boundingRect(contur_qr)
    qr_decupat = image[y:y + h, x:x + w]

    #Versiune QR =?
    h,w = qr_decupat.shape
    primul_rand = qr_decupat[0]
    curent = primul_rand[0]
    count = 0
    for pixel in primul_rand:
        if pixel == curent:
            count+=1
        else:
            break
    dim_modul = count//7
    versiune = (w //dim_modul - 21)//4 +1

    #construiesc matrice binara
    nr_linii = nr_coloane = 21 + 4*(versiune-1)
    M = np.zeros((nr_linii, nr_coloane), dtype=int)
    for i in range(nr_linii):
        for j in range(nr_coloane):
            y_start, y_end = i * dim_modul, (i + 1) * dim_modul
            x_start, x_end = j * dim_modul, (j + 1) * dim_modul
            modul = qr_decupat[y_start:y_end, x_start:x_end]
            medie = np.mean(modul)
            if medie<128:
                M[i, j] = 1

    #ce masca s-a folosit?
    masca = 4*M[8][2] + 2*M[8][3] + M[8][4]

    #generez lista de elemente ce nu trebuie sa fie afectate de masca
    excluded_coords = []
    final = 21 + 4*(versiune-1)
    #Pozitionare+separator:
    pozitii = [(0,0), (final-8,0),(0,final-8)]
    for poz in pozitii:
        x,y = poz
        for row in range(x,x+8):
            for col in range(y,y+8):
                excluded_coords.append((row,col))
    #sincronizare:
    for i in range(final):
        excluded_coords.append((6, i))  #orizontală
        excluded_coords.append((i, 6))  #verticală
    #format
    for i in range(9):
        excluded_coords.append((8,i))
        excluded_coords.append((i,8))
        if final-i<29:
            excluded_coords.append((8, final-i))
            excluded_coords.append((final - i,8))
    #aliniere, dictionar de valori predefinite centru
    alig = {2:(18,18),3:(22,22),4:(26,26),5:(30,30),6:(34,34)}
    if versiune>1:
        centru = alig[versiune]
        ax,ay = centru
        for row in range(ax-2,ax+3):
            for col in range(ay-2,ay+3):
                excluded_coords.append((row,col))
    #aplicare masca
    final = 21+ (versiune-1)*4
    if masca==0:
        for row in range(final):
            for col in range(final):
                if (row,col) not in excluded_coords:
                    masca0(M,row,col)
    elif masca==1:
        for row in range(final):
            for col in range(final):
                if (row,col) not in excluded_coords:
                    masca1(M,row,col)
    elif masca==2:
        for row in range(final):
            for col in range(final):
                if (row,col) not in excluded_coords:
                    masca2(M,row,col)
    elif masca==3:
        for row in range(final):
            for col in range(final):
                if (row,col) not in excluded_coords:
                    masca3(M,row,col)
    elif masca==4:
        for row in range(final):
            for col in range(final):
                if (row,col) not in excluded_coords:
                    masca4(M,row,col)
    elif masca==5:
        for row in range(final):
            for col in range(final):
                if (row,col) not in excluded_coords:
                    masca5(M,row,col)
    elif masca==6:
        for row in range(final):
            for col in range(final):
                if (row,col) not in excluded_coords:
                    masca6(M,row,col)
    elif masca==7:
        for row in range(final):
            for col in range(final):
                if (row,col) not in excluded_coords:
                    masca7(M,row,col)
    #M=aplica_masca(M,masca) #Nu vrea asa
    #decodare
    bits = []
    rows, cols = len(M), len(M[0])

    col = cols - 1  # Începem din colțul din dreapta jos
    row = rows - 1
    direction = -1  # -1 pentru sus, 1 pentru jos

    while col >= 0:
        if col == 6:  # Coloana 6 este rezervată pentru pattern de sincronizare
            col -= 1

        while 0 <= row < rows:
            if (row, col) not in excluded_coords:  # Ignoră zonele fixe
                bits.append(M[row][col])
            if(row,col-1) not in excluded_coords:
                bits.append(M[row][col-1])
            row += direction  # Ne deplasăm sus sau jos

        direction *= -1  # Schimbăm direcția
        col -= 2  # Ne mutăm la următoarele două coloane din stânga
        row +=direction
    #codurile sunt byte mode, incep cu 0100
    #urmatorii 8 biti sunt lungimea
    length = 128 * bits[4] +64*bits[5]+32*bits[6]+16*bits[7]+8*bits[8]+4*bits[9]+2*bits[10]+bits[11]
    sol=''
    i=12
    while length>0:
        if i+8>len(bits):
            break
        byte_value = int(''.join(map(str, bits[i:i + 8])), 2)
        sol += chr(byte_value)
        i+=8
        length-=1
    print(sol)
else:
    print("Input invalid")
