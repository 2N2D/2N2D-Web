# 4 2N2D API

**2N2D API** este componenta aplicației 2N2D responsabilă pentru procesarea modelelor de rețele neuronale și a seturilor de date.

- Este dezvoltat în **Python**.
- Rulează pe un server **FastAPI**, care gestionează cererile primite de la partea web a aplicației.

**Structura principală a API-ului include:**

- **Endpoint-uri REST**: pentru operațiuni precum încărcarea modelelor, optimizarea acestora, analiza datelor CSV, generarea de rezultate etc.
- **Mecanisme de autentificare și validare**: verificarea cheii API, asocierea sesiunii, validarea datelor primite.
- **Managementul fișierelor**: descărcarea și încărcarea fișierelor din cloud (R2 Bucket).
- **Procese de optimizare**: implementarea algoritmilor de optimizare a rețelelor neuronale (Brute Force, Genetic, Neuro-evolutiv).
- **Procesare internă**: module Python (2N2D.py, data.py) care realizează analiza și procesarea efectivă a modelelor și datelor.
- **Format de răspuns**: datele procesate sunt returnate clientului în format JSON.

## 4.1 **APP.py**

în acest fișier rulează serverul și deci întregul API, aici ajunge și este procesat fiecare request din partea WEB a aplicației.

### 4.1.1 Autentificare

Toate endpoint-urile API necesită transmiterea unui session-id în antetul (header-ul) cererii. Acest ID este utilizat pentru a asocia cererile cu o sesiune specifică de utilizator.

- **Antet (Header):** session-id
- **Valoare:** Un identificator unic pentru sesiunea utilizatorului, obținut prin getCurrentUser() sau getSessionTokenHash() din biblioteca de autentificare a frontend-ului.

De asemenea toate acestea au nevoie și de un API key

- **Antet:** x-api-key
- **Valoare:** Un hash folosind sha256 a cheii API, configurată la startul aplicației de către deployer. Lipsa acestei chei va rezulta în refuzarea cererii de către API.

### 4.1.2 Exemplu de Antet

\\{
"x-api-key":"hash-cheie-api",  
"session-id": "hash-sesiune-utilizator"  
\\}

##

## 4.2 **Endpoint-uri**

### 4.2.1 **Verificare Stare (Health Check)**

#### **GET /**

Acest endpoint poate fi folosit pentru a verifica dacă API-ul rulează.

- **Descriere:** Confirmă că API-ul este activ.
- **Corp Cerere (Request Body):** Niciunul

**Răspuns de Succes (200):**

\\{  
"message": "2N2D API is running"  
\\}

### 4.2.2 **Gestionarea Datelor (Data Handling)**

Aceste endpoint-uri sunt responsabile pentru încărcarea și procesarea datelor necesare pentru optimizarea modelului.

#### **POST /upload-model**

Încarcă un fișier model în format ONNX pentru analiză.

- **Descriere:** Frontend-ul trimite calea fișierului model, iar serverul încarcă modelul ONNX de la acea cale. Metadatele modelului sunt extrase și returnate.

**Corp Cerere:**

\\{  
"filepath": "cale/catre/modelul/tau.onnx"  
\\}

**Răspuns de Succes (200):**

\\{  
"input_names": \\["input1", "input2"\\],  
"output_names": \\["output"\\],  
"layers": \\[
\\{ "name": "layer1", "type": "Conv", "attributes": "..." \\},
\\{ "name": "layer2", "type": "ReLU", "attributes": "..." \\}
\\]  
\\}

#### **POST /upload-csv**

Încarcă un fișier CSV care conține setul de date.

- **Descriere:** Frontend-ul furnizează calea către un fișier CSV. Serverul încarcă acest fișier, extrage antetele (coloanele) și o previzualizare a datelor, și returnează aceste informații.

**Corp Cerere:**

\\{  
"filepath": "cale/catre/datele/tale.csv"  
\\}

**Răspuns de Succes (200):**

\\{  
"columns": \\["feature1", "feature2", "target"\\],  
"preview": \\[
\\{ "feature1": 0.5, "feature2": 0.3, "target": 1 \\},
\\{ "feature1": 0.6, "feature2": 0.4, "target": 0 \\}
\\]  
\\}

### 4.2.3 **Optimizarea Modelului**

Acest set de endpoint-uri gestionează procesul de optimizare a modelului.

#### **POST /optimize**

Inițiază procesul de optimizare a modelului.

- **Descriere:** Acest endpoint pornește căutarea arhitecturii rețelei neuronale. Necesită caracteristicile de intrare (input features), variabila țintă (target) și alți parametri de configurare. Procesul rulează în fundal.

**Corp Cerere:**

\\{  
"input_features": \\["feature1", "feature2"\\],  
"target_feature": "target",  
"max_epochs": 100,  
"session_id": "id-sesiune-utilizator",  
"csv_path": "cale/catre/date.csv",  
"onnx_path": "cale/catre/model.onnx",  
"encoding": "onehot",  
"strategy": "bayesian"  
\\}

**Răspuns de Succes (200):**

\\{  
"message": "Optimizarea a început cu succes",  
"url": "id-sesiune-utilizator/optim/model_optimizat.onnx",  
"model_path": "/cale/server/catre/model_optimizat.onnx",  
"results": \\{  
"best_accuracy": 0.95,  
"best_hyperparameters": "..."  
\\}  
\\}

**Răspuns de Eroare (400):**

\\{  
"detail": "Mesaj de eroare care detaliază ce nu a funcționat în timpul optimizării"  
\\}

#### **GET /optimization-status/{session_id}**

Transmite în flux (streams) starea procesului de optimizare în desfășurare.

- **Descriere:** Acesta este un endpoint de tip Server-Sent Events (SSE) care permite frontend-ului să primească actualizări în timp real despre progresul optimizării pentru o sesiune dată.
- **Parametri URL:**
  - session_id: ID-ul sesiunii pentru care se dorește starea.

**Răspuns (text/event-stream):** Un flux de obiecte JSON cu actualizări de stare.

data: \\{"status": "Se descarcă datele CSV din baza de date...", "progress": 3\\}

data: \\{"status": "Se codează datele CSV...", "progress": 5\\}  
data: \\{"status": "Se caută arhitectura optimă...", "progress": 20\\}

## 4.3 Request flow

Pentru fiecare request primit, API-ul urmează următorii pași:

1. Se verifică hash-ul cheii API, comparându-se cu cel primit în header-ul requestului.
2. Se asociază procesul cu session-id-ul aferent.
3. Se extrag datele din corpul (body) requestului.
4. Se descarcă fișierele CSV și ONNX din R2 Bucket, pe baza ID-ului primit.
5. Toate datele sunt trimise către modulele interne 2N2D.py sau data.py pentru procesare.
6. Răspunsurile generate sunt formatate în JSON și trimise înapoi clientului care a făcut cererea.

## 4.4 Vizualizare

Structura modelelor încărcate este extrasă automat folosind librăria ONNX. Fiecare element al rețelei este reprezentat grafic sub forma unui nod, căruia i se atribuie:

- **ID unic**
- **Label** — numele afișat utilizatorului (de exemplu: LSTM, RELU, Transpose etc.)
- **Titlu** — o descriere detaliată a nodului
- **Grup** — categoria din care face parte nodul (de ex. funcții de activare, operații de transformare etc.)

Secvența de aplicare a operațiilor este păstrată fidel, pentru a reflecta corect fluxul datelor prin rețea.

Toate aceste informații sunt apoi serializate într-un format JSON și transmise către frontend, unde sunt utilizate pentru reprezentarea vizuală și interactivă a modelului.

## 4.5 Analiza Fișierelor ONNX

2N2D implementează o analiză completă și automată a fișierelor ONNX încărcate de utilizator, asigurând o înțelegere detaliată a arhitecturii rețelelor neuronale. Această analiză cuprinde următoarele funcționalități:

- **Suport pentru opset-uri multiple:** Analiza este compatibilă cu diferite versiuni ale setului de operații (opset), asigurând flexibilitate în procesarea modelelor.
- **Extracția dimensiunilor tensorilor:** Se preiau dimensiunile exacte ale tensorilor folosiți în rețea pentru o reconstrucție fidelă.
- **Măsurarea frecvenței și ordinii operațiilor:** Se numără aparițiile fiecărei operații și se păstrează succesiunea acestora pentru analiza fluxului de date.
- **Detectarea categoriilor principale de arhitecturi:** Se identifică tipurile majore de rețele, cum ar fi Convolutionale (Conv), LSTM sau FeedForward, inclusiv subtipurile acestora (Conv1D, Conv2D, GRU, RNN).
- **Identificarea elementelor specifice fiecărei arhitecturi:**
  - Pentru rețele convoluționale, sunt detectate operațiile de pooling și filtrele utilizate.
  - Pentru rețelele LSTM, se analizează direcționalitatea și alte caracteristici specifice.
- **Recunoașterea operațiilor de transformare:** Operații precum Transpose, Slice, Reshape, Flatten sunt identificate corect și validate pentru a asigura utilizarea lor adecvată în rețea.
- **Detectarea funcțiilor de activare:** Sunt recunoscute funcții precum RELU, Softmax, Sigmoid și altele, esențiale pentru dinamica rețelei neuronale.

Prin combinarea acestor elemente, 2N2D poate infera arhitectura completă a modelului ONNX, facilitând astfel reconstruirea lui în PyTorch — un pas crucial în procesul avansat de optimizare oferit utilizatorilor.

## 4.6 Analiza Fișierelor CSV

Fișierele CSV reprezintă cel mai comun format pentru date tabulare utilizate în Machine Learning. Aceste fișiere pot conține tipuri variate de date, cum ar fi numere, valori booleene sau text. Totuși, rețelele neuronale lucrează doar cu date numerice, ceea ce face necesară transformarea (encodarea) datelor non-numerice în valori numerice.

### 4.6.1 Metode de encodare

2N2D oferă două metode principale pentru encodarea datelor non-numerice:

#### 1\. Encodare Label

- Fiecare valoare non-numerică unică dintr-o coloană primește un cod numeric unic, începând de la 0.
- Exemplu:  
   Categorii ordonate, precum ("ușor", "mediu", "dificil") vor fi encodate ca (0, 1, 2).
- Această metodă este ideală atunci când există o succesiune logică între elemente, permițând rețelei să interpreteze valorile ca având o relație ordonată.
- Nu este potrivită pentru categorii fără ordine logică, de exemplu ("albastru", "galben", "roșu"), deoarece codificarea (0, 1, 2) poate induce rețeaua în eroare, sugerând o ierarhie inexistenta.
- Este o metodă eficientă din punct de vedere al memoriei.

#### 2\. Encodare One-Hot

- Pentru fiecare element unic non-numeric se creează un vector binar cu dimensiunea egală cu numărul de valori unice.
- Exemplu:

Valorile ("albastru", "galben", "roșu") vor fi encodate astfel:  
"albastru" → \[1, 0, 0\]  
"galben" → \[0, 1, 0\]  
"roșu" → \[0, 0, 1\]

- Această metodă elimină problema ordinii false de la encodarea Label, deoarece fiecare categorie este reprezentată distinct și independent.
- Dezavantaj: vectorii pot deveni foarte mari dacă există multe valori unice, crescând substanțial consumul de memorie.

### 4.6.2 Protecția utilizatorului împotriva consumului excesiv de memorie

- 2N2D calculează automat numărul de elemente unice din coloanele non-numerice și estimează dimensiunea fișierului după aplicarea encodării One-Hot.
- Dacă această estimare indică o creștere semnificativă a consumului de memorie, utilizatorul este avertizat printr-un mesaj clar.
- Alerta include și informații utile, precum coloanele care contribuie cel mai mult la această creștere a dimensiunii.

### 4.6.3 Vizualizare și analiză a datelor CSV

Pe lângă encoding, utilizatorul poate:

- Vizualiza grafice bazate pe datele din fișier, cum ar fi heatmaps și distribuții ale valorilor.
- Consulta un rezumat statistic care prezintă datele principale din fișier (media, deviația standard, valori minime și maxime etc.).

## 4.7 Optimizarea Rețelelor Neuronale

API-ul utilizează un proces complex de optimizare a rețelelor neuronale, structurat în următoarele etape:

### 4.7.1. Prelucrarea datelor

- Datele sunt validate și separate în seturi pentru antrenare și testare.
- Se realizează prelucrarea acestora și, dacă este disponibil, datele sunt încărcate pe GPU pentru procesare mai rapidă.

### 4.7.2. Analiza modelului ONNX

- Fișierele ONNX încărcate sunt analizate în detaliu.
- Modelul original este reprodus în PyTorch pentru o manipulare și reantrenare ulterioară.

### 4.7.3. Reantrenarea modelului original

- O copie a modelului original este reantrenată și evaluată.
- Acest pas oferă o bază de referință a performanței inițiale, care va fi folosită ulterior pentru comparații.

##

##

##

##

##

## 4.8 Metodele de optimizare

De aici, procesul diferă în funcție de metoda de optimizare aleasă, după cum urmează:

### 4.8.1 Brute Force

- Modelul este reantrenat și testat pe o varietate de configurații.
- Numărul de neuroni per layer variază în intervalul \[x/2, x\*2\], unde x este numărul original de neuroni.
- Numărul de layere este variat în intervalul \[y-1, y+1\], y fiind numărul original de layere.
- Antrenarea folosește tehnica _early stopping_ pentru a opri procesul dacă modelul nu mai prezintă îmbunătățiri.
- După testarea tuturor configurațiilor, cel mai performant model este salvat și oferit utilizatorului împreună cu metricile de evaluare (descrise în User Manual).

### 4.8.2 Genetic (folosind librăria DEAP)

- Se creează indivizi care au două atribute:
  - **enhancement factor** — cât de mult este modificat modelul original
  - **pattern** — tipul modificării aplicate (fără modificări speciale, modificări în lățime (width), în adâncime (depth) sau ambele)
- Modelele sunt recreate și reantrenate pe baza caracteristicilor acestor indivizi.
- Modificările nu mai sunt restrânse la intervalele fixe, ci sunt determinate de atributele indivizilor.
- Indivizii cu performanță superioară supraviețuiesc și se recombină, generând generații noi care suferă mutații pentru diversitate.
- Procesul continuă până la obținerea celui mai performant model, care este oferit utilizatorului.
- Această metodă este mai costisitoare computațional și în timp față de Brute Force, dar explorează un spațiu mai larg de configurații.

###

###

###

###

###

### 4.8.3 Neuro-evolutiv (folosind librăria NEAT)

- Pornește de la o populație simplă de genomuri ce descriu structura rețelei neuronale.
- Un genom este compus din:
  - Noduri (cu ID, bias, factor de răspuns, funcție de activare și tip de agregare)
  - Conexiuni (cu ID, greutate și starea activă/inactivă)
  - Fitness (scorul de performanță)

- După evaluarea generației inițiale, genomurile sunt grupate în specii în funcție de similaritate, iar cele mai bune au șanse mai mari de recombinare.
- Genomurile rezultate suferă mutații și sunt evaluate iar și iar în cicluri repetate.
- Această metodă permite modificarea atât a greutăților și biasurilor, cât și a topologiei rețelei, oferind arhitecturi complet noi și complexe.
- Modelul final descris de cel mai bun genom este returnat.

### 4.8.4 Finalizarea procesului

Modelul optimizat rezultat din oricare dintre metodele de mai sus este antrenat și evaluat încă o dată pentru a confirma performanța. Ulterior, este exportat în format ONNX și pus la dispoziția utilizatorului pentru instalare și utilizare.![]()

## 4.9 Scalabilitate

Datorită faptului că API-ul este împărțit și structurat în diferite clase și fișiere, crearea unei noi funcționalități este foarte simplă. Tot ce trebuie făcut este:

- implementarea funcționalității într-un nou fișier;
- crearea unui nou endpoint în APP.py.

De asemenea, datorită centralizării fișierelor și a datelor utilizatorilor, este posibilă crearea mai multor instanțe ale API-ului și împărțirea load-ului între acestea.

## 4.10 Securitate

Toate requesturile care ajung în API sunt mai întâi verificate. Se extrag datele din header și se caută componenta x*api_key. Aceasta este, de fapt, un \_hash* al unei chei stocate pe serverul aplicației web.

Hash-ul primit este comparat cu hash-ul stocat local, setat la pornirea aplicației. API-ul nu cunoaște cheia originală. Dacă hash-urile se potrivesc, atunci requestul este procesat; dacă nu, acesta este refuzat.

În acest fel putem filtra requesturile în cazul în care endpoint-ul API-ului este descoperit. Totuși, pentru a ne asigura că endpoint-ul rămâne ascuns, orice request către API din interfața web pleacă **doar din server**, și **nu din client**.

Astfel, serverul poate refuza complet requesturile neautorizate, facilitând protecția împotriva atacurilor de tip **DoS** (_Denial of Service_).

De asemenea, configurarea serverului care rulează aplicația este foarte importantă și ar trebui să includă, în configurația sa, măcar elementele de bază, precum un **firewall**.

## 4.11 Tehnologii folosite

\- Pytorch - Framework principal pentru rețele neuronale

\- ONNX - Format standard al modelelor de machine learning

\- Numpy - Procesare numerică și manipulare a datelor

\- Pandas - Preprocesare a datelor tabulare

\- Scikit-learn - Funcții de preprocesare și evaluare

\- NEAT - Algoritm neuro-evolutiv folosit pentru optimizare

\- DEAP - Algoritm genetic folosit pentru optimizare

\- FastAPI - Framework pentru API

\- Cloudfare R2 Buckets - Stocare fisiere
