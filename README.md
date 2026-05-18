# Evolving Depths - Prototipo IA + Pygame

Prototipo giocabile che integra:
- Multi-Armed Bandit (DEA, epsilon-greedy) per il pacing della difficolta
- Search-based PCG (automata cellulare) per la generazione stanza
- Verificatore A* per la giocabilita della mappa
- Logging CSV e runtime debug per validare comportamento dei componenti

Per una spiegazione completa di regole, HP, goal, stanze, power-up, nemici, eventi e sistemi avanzati, vedi [GAME_GUIDE.md](GAME_GUIDE.md).

## Struttura

- `main.py`: loop Pygame 60 FPS, input, update, draw, transizione stanza
- `config.py`: costanti globali, inclusa `USE_BANDIT`
- `ai/bandit_dea.py`: scelta difficolta via epsilon-greedy
- `ai/generator.py`: generazione mappa candidata
- `ai/verifier.py`: verifica raggiungibilita ingresso -> uscita con A*
- `game_logic/director.py`: orchestrazione Observe -> Decide -> Apply -> Verify
- `game_logic/observer.py`: metriche stanza e log CSV
- `game_logic/runtime_logger.py`: logger eventi runtime per debug dei componenti
- `logs/session_metrics.csv`: output logging con colonne richieste
- `logs/engagement_metrics.csv`: metriche engagement estese per stanza
- `logs/runtime_debug.log`: eventi runtime (main, director, observer)

## Gameplay (in breve)

- Obiettivo principale: esplorare un dungeon di stanze interconnesse e raccogliere reliquie.
- Obiettivo finale: quando raccogli `RELIC_GOAL` reliquie (default 3), si apre la porta finale nella stanza finale.
- La stanza finale e la piu lontana dalla stanza iniziale nel grafo dungeon.
- Le porte verdi sui bordi collegano stanze adiacenti del dungeon.
- Sconfitta: HP a zero.
- I nemici rossi inseguono il player: si muovono con pathfinding su griglia.
- Velocita nemici adattiva al flow state del player:
   - flow alto -> nemici piu rapidi
   - flow basso -> nemici meno aggressivi
- Le stanze non hanno un solo layout: il generatore alterna profili `cave`, `arena`, `corridor`,
   `pillars`, `crossroads`, `islands` con rotazioni/specchiature casuali e densita ostacoli differenti.
- Il profilo stanza ora pesa anche l'engagement:
   - engagement basso -> maggiore probabilita di `arena` e stanze piu leggibili
   - engagement medio -> piu probabilita di `pillars` e `islands`
   - engagement alto -> maggiore probabilita di `corridor` e `crossroads`, quindi pressione piu alta
- Ogni porta ha un marker di spawn visibile subito all'interno della stanza:
   quando attraversi una connessione, entri esattamente su quel marker.
- Alcune stanze propongono un mini-obiettivo opzionale: ora non blocca piu l'uscita, ma se lo completi ottieni una piccola ricompensa.
- Esistono stanze elite, stanze merchant, stanze sanctuary e stanze rischio/ricompensa.
- Il numero di nemici nasce dall'engagement e anche la comparsa di stanze speciali/elite ora e guidata dall'engagement corrente.
- Il dungeon puo attivare eventi globali come blackout, infestazione e predator hunt.
- I nemici hanno ruoli diversi (`stalker`, `charger`, `ranged`, `blocker`, `splitter`, `predator`).
- Le combo di flow possono sbloccare reward room e bonus temporanei.
- Alcune stanze cambiano quando ci torni: il dungeon ha memoria locale.

Il flow viene stimato da tempo di completamento e danno subito.

## Setup

1. Attiva ambiente virtuale:
   - `source ./venv/bin/activate`
2. Installa dipendenze:
   - `pip install -r requirements.txt`
3. Avvia il gioco:
   - `python main.py`

### Modalita test rapido

- Test automatico componenti senza input utente:
   - `python main.py --self-test`
- Avvio main loop con auto-stop (utile in CI/headless):
   - `SDL_VIDEODRIVER=dummy python main.py --max-frames 120`
- Per forzare HUD testuale (disabilitato di default):
   - `AIE_ENABLE_FONT=1 python main.py`

## Controlli

- Frecce direzionali: movimento
- Usa le porte verdi per spostarti tra stanze collegate
- Entrare in una cella Enemy elimina quel nemico ma puo infliggere danno da contatto

## Power-up

- Verde (Heal): recupero HP. In HUD e nella stanza usa icona con croce.
- Blu (Speed): movimento potenziato per N step. In HUD e nella stanza usa icona a freccia.
- Viola (Shield): blocca i prossimi colpi nemici. In HUD e nella stanza usa icona a scudo.
- Teleport A/B: due zone collegate che trasferiscono istantaneamente il player
- Combo: `Speed + Shield` attiva un protected dash
- Combo: `Heal + Relic` puo generare overheal temporaneo

La HUD mostra anche un pannello dedicato ai potenziamenti:
- `Heal`: effetto del pickup
- `Speed`: step di boost ancora attivi
- `Shield`: colpi ancora bloccabili

Quando `Speed` o `Shield` sono attivi, il relativo riquadro viene evidenziato.

## Legenda Colori

- Giallo: player
- Rosso: nemico
- Verde acceso: porta stanza (transizione)
- Bianco brillante: reliquia
- Azzurro: spawn point vicino a una porta
- Rosso scuro: porta finale bloccata
- Verde chiaro: porta finale aperta
- Verde orb: Heal
- Blu orb: Speed
- Viola orb: Shield
- Ciano + Oro: coppia teleport A/B

## HUD

La HUD mostra:
- Numero stanza corrente (ID stanza dungeon)
- Barra HP
- Barra overheal se presente
- Barra reliquie raccolte (`collected/goal`)
- Barra Flow
- Pannello potenziamenti con icone e stato corrente
- Obiettivo stanza corrente, se presente
- Evento globale del dungeon
- Combo streak corrente
- Modificatore elite della stanza, se presente
- Minimappa del dungeon con stanza corrente, connessioni, stanza finale e stanze reliquia scoperte

Se il modulo font non e disponibile, la HUD resta grafica e i dettagli testuali sono visibili nel titolo finestra.

## Ablation

In `config.py`:
- `USE_BANDIT = True` -> modalita adaptive (Bandit attivo)
- `USE_BANDIT = False` -> baseline random (Bandit disattivo)

## Log per valutazione

Ad ogni stanza completata viene scritto un record:
- `id_sessione, num_stanza, use_bandit, difficulty_chosen, time_taken, hp_lost`

File log: `logs/session_metrics.csv`

File log engagement esteso: `logs/engagement_metrics.csv`

Colonne principali engagement:
- dungeon_room_id
- flow_score, enemy_delay
- enemy_spawned, enemy_kills, enemy_hits_taken, damage_blocked
- powerups_spawned, powerups_collected, heal_collected, speed_collected, shield_collected
- teleport_present, teleport_uses
- moves_made, avg_enemy_distance
- room_profile
- relics_collected, relic_goal, final_door_open

Log diagnostico runtime: `logs/runtime_debug.log`

Nel runtime log trovi anche:
- `enemy_attack` per attacchi nemici
- `features_spawned` / `SELF_TEST features` per spawn power-up e teleport
- `room_generated` / `room_completed` per transizioni stanza
