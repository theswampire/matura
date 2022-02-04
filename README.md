# Maturitätsarbeit

Dieses Repository enthält die Maturitätsarbeit 2022 von Kai Siegfried: 
**Training eines wachsenden neuronalen Netzwerks — Herleitung und Implementation eines Neuronen-Teilungsverfahrens**.

Dazu gehört:
- `Maturitätsarbeit_2022_Kai_Siegfried.pdf`
- *Python* Framework `kAI`
    - Beispiele unter `examples`


---


## Framework `kAI`

`kAI` ist ein Python-Framework für neuronale Netzwerke und ist das Produkt der Maturitätsarbeit. Damit können nebst den konventionellen Feedforward-Netzwerken auch wachsende neuronale Netzwerke mithilfe eines Neuronen-Teilungsverfahrens trainiert werden.

## Anleitung `kAI`

Das Framework `kAI` wurde mit **Python 3.10** auf **Windows** entwickelt und getestet. Allerdings sollte `kAI` ebenfalls auf Unix funktionieren.

Folgende Packages werden benötigt:
- `numpy`
- `matplotlib`
- `colorama`

Mit dem folgenden Befehl können die Abhängigkeiten installiert werden:
```console
pip install -r requirements.txt
```

### Beispiele

Im Ordner `examples` sind einige Beispiele abgelegt, die unter anderem für die Untersuchungen während der Arbeit verwendet wurden. 
Mit diesen Beispielen sollte klar werden, wie diese Software zu gebrauchen ist.

Falls der Fehler `ModuleNotFoundError: No module named 'kAI'` auftaucht, gibt es zwei Möglichkeiten zur Behebung:

1.  Das auszuführende Programm und den Ordner `kAI` in den gleichen Ordner verschieben.
2.  Den Pfad zu `kAI` dem `sys.path` hinzufügen. Dazu kann folgendes an den Anfang des Skripts hinzufügen:
```python
import sys
# Falls der Order 'kAI' unter 'C:\\Users\Benutzername\Documents\matura' liegt
sys.path.append(r"C:\\Users\Benutzername\Documents\matura")

# Der Rest des Programms
# ... 
```

---

## Anmerkung

Das Repository dient nur zur Veröffentlichung des Quellcodes der Arbeit.
Fehler und Probleme können selbstverständlich unter **Issues** gemeldet werden, jedoch besteht keine Gewähr, dass diese bearbeitet werden.

`kAI` wird hier nicht aktiv weiterentwickelt und unterhalten.
Allerdings besteht die Möglichkeit, dass eine überarbeitete Version des Frameworks entwickelt wird, die leistungsorientiert und "production-ready" sein wird.