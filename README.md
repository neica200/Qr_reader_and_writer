# Qr_reader_and_writer
#Numele echipei este BitScanner:
  Neica Mario-Alexandru(134),Roibu Amelia-Maria(134) Danila Tudor-Mihail(134), Draghici Bianca-Elena(134)
#Descriere a programului
  Cand programul se initializeaza userul are de ales intre modul de citire si modul de generare(1 sau 2)
  ![image](https://github.com/user-attachments/assets/e2706e36-21f1-4551-b5aa-ca24bcc639af)

  -Daca acesta alege modul de generare a unui cod QR trebuie apoi sa introduca un sir de caractere. Programul ia sirul de caractere, ii calculeaza lungimea si apoi transforma fiecare caracter intr-o       
   secventa de 8 biti, corespunzatoare codului ASCII. Sirul pe care il avem acum contine pe primele 4 pozitii 0100, pt byte mode, pe urmatorii 8 biti lungimea sirului,datele utilizatorului si bitii de 
   padding. Se imparte pe blocuri de cate 8 biti(codewords) si cu ajutorul algoritmului Reed-Solomon se genereaza portiunile care ajuta la repararea erorilor(Noi folosim L-mode, care recupereaza pana la 7%     din date).Dupa aceea creeaza o matrice binara(de 0 si 1), in care pune mai intai finder patterns, separatorii, timing patterns si daca e nevoie(versiunea 2 in sus) allignment pattern-ul. Se marcheaza        aceste zone,pentru a nu fi alterate, iar apoi, pornind de pe ultima casuta din coltul dreapta jos, se adauga modul,lungimea,datele si codewordsurile de corectare, intr-un pattern care merge in zig-zag,      de jos in sus si alternativ de sus in jos. Se aplica apoi toate cele 8 masti, se calculeaza pentru fiecare scorul de penalizare si se alege masca cu cel mai mic scor. Intr-un final, adaugam codurile   
   de format, care contine 01(Modul L), urmatorii 3 biti care reprezinta masca folosita si apoi 10 biti ce ajuta la codificarea BCH. Cu ajutorul bibliotecii matplotlib.pyplot generam un png care contine         codul qr dorit.
   Exemplu pentru numele echipei:
 CodQr_generat poza

  -Daca acesta alege modul de citire, trebuie sa introduca numele qr-lui din fisier pe care acesta doreste sa le scaneze. Noi am folosit cele doua coduri qr generate online, pentru numele echipei si link-ul   paginii de curs:
  -cele doua coduri qr
  Cu ajutorul bibliotecii cv2, eliminam marginile albe din poza(binarizam si eliminam fundalul) pentru a ramane doar cu codul qr, calculam marimea unui modul numarand numarul de pixeli negrii de primul    rand, dimensiunea unui modul fiind acest numar impartit la 7(un finder pattern are mereu 7 module lungime). Stiind dimensiunea unui modul, impartim la lungimea pozei si determinam versiunea codului qr.Stiind versiunea, generam o matrice si o completam corespunzator mediei din patratul determinat de dimensiunea unui modul si astfel obtinem o matrice 1 la 1 cu codul qr.Facem o lista cu coordonatele care contin elementele importante din cod, in functie de versiune, pentru a nu le distorsiona cu masca si a nu le lua in calcul cand citim bitii. Ne uitam in partea de format si aflam ce masca s-a folosit, scoatem masca dupa formula specifica. Ne uitam la primii patru biti(Orientarea e tot de la coltul din dreapta jos, de jos in sus) si verificam modul de encodare, iar urmatorii 8 biti ne spun lungimea mesajului. Deobicei e byte mode, asa ca incepem sa luam cate 8 biti si sa generam caracterul corespunzator codului ascii scris pe acei 8 biti.

