# Soluzione dell'Equazione del Calore utilizzando Physics-Informed Neural Networks (PINNs)

Questo repository contiene il codice per risolvere l'equazione del calore utilizzando una rete neurale informata dalla fisica (PINN) con vincoli rigidi. I risultati mostrano una buona corrispondenza tra la soluzione predetta dalla rete neurale e la soluzione analitica, dimostrando l'efficacia dell'approccio hPINN per problemi di questo tipo.

## Descrizione

L'equazione del calore è una PDE fondamentale utilizzata per modellare la diffusione del calore in un materiale. In questo progetto, è stata applicata una PINN per risolvere questa equazione, integrando le leggi fisiche direttamente nella struttura della rete neurale. 

## Dettagli dell'Implementazione

Il codice utilizza la libreria `DeepXDE` (https://deepxde.readthedocs.io/en) per definire e addestrare la PINN. Di seguito sono riportati i passaggi principali dell'implementazione:

1. **Impostazione del Seed**: Per garantire la riproducibilità dei risultati.
2. **Definizione della PDE**: Utilizzo dell'auto-differenziazione per calcolare il termine dell'errore della PDE.
3. **Soluzione Analitica**: Utilizzata per confrontare i risultati predetti dalla rete neurale.
4. **Definizione del Dominio e delle Condizioni Iniziali/di Contorno**: Definizione del problema fisico.
5. **Definizione della Rete Neurale**: Utilizzo di una rete feed-forward con 3 strati da 12 neuroni ciascuno.
6. **Metodo delle Penalità**: Implementazione di un ciclo iterativo per aggiornare i coefficienti di penalità e addestrare il modello.
7. **Salvataggio e Valutazione del Modello**: Salvataggio del modello finale e confronto con la soluzione analitica.
8. **Visualizzazione dei Risultati**: Generazione di grafici per visualizzare i risultati predetti e i loss medi durante l'addestramento.

## Prerequisiti

Assicurarsi di avere installato le seguenti librerie Python:
- `numpy`
- `deepxde`
- `matplotlib`

È possibile installarle utilizzando `pip`:
```bash
pip install numpy deepxde matplotlib
```
## Utilizzo

1. **Clonare il repository**
```bash
git clone https://github.com/luliannovi/hPINN
cd repository
```
2. **Eseguire lo script principale:**
```bash
python v4.py
```

## Visualizzazione confronto con soluzione veritiera
```bash
python visualize_results.py
```
