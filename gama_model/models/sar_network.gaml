/**
 * SAR Network — Visualización 3D en GAMA GUI con datos en tiempo real.
 *
 * Abre este modelo en GAMA Platform (GUI) y ejecuta el experimento.
 * Se conecta como cliente TCP a un servidor Python que ejecuta la simulación.
 * Python controla toda la lógica; GAMA solo renderiza y anima.
 *
 * Uso:
 *   1. python examples/02_swarm/03_gama_visualization.py --scenario 1
 *   2. Doble-click en "Experiment sar_gui_network" en GAMA
 *   3. Pulsa (Play) en la barra de herramientas de GAMA
 *
 * Protocolo TCP (campos separados por |, cada comando termina en ~\n):
 *   INIT|nd|ndog|nv|cols|rows
 *   DRONE|idx|x|y|budget   DOG|idx|x|y|budget   VICTIM|idx|x|y
 *   INIT_END   TICK|step   AGENT|type|idx|x|y|budget|active
 *   FOUND|x|y   TICK_END   LINKS|n|x1,y1,x2,y2;...   END
 */
model SARNetwork

global skills: [network] {
    // --- Parámetros configurables desde la GUI ---
    string python_host <- "localhost";
    int python_port <- 6869;

    // --- Datos estáticos ---
    file heatmap_file <- csv_file("../includes/heatmap.csv");

    // --- Fields para overlay ---
    // prob_matrix conserva la resolución completa: de ella se derivan
    // grid_cols/grid_rows (tamaño del mundo y mapeo de coordenadas).
    matrix prob_matrix <- matrix(heatmap_file);
    int grid_cols <- prob_matrix.columns;
    int grid_rows <- prob_matrix.rows;
    // Heatmap renderizado a resolución reducida (prob_display_matrix, rellenada
    // en init muestreando prob_matrix) para abaratar su único render. prob_ds: ↑ más óptimo.
    int prob_ds <- 2;
    int disp_cols <- max(1, int(grid_cols / prob_ds));
    int disp_rows <- max(1, int(grid_rows / prob_ds));
    matrix prob_display_matrix <- 0.0 as_matrix {disp_cols, disp_rows};
    field probability_field <- field(prob_display_matrix);
    // Overlay de exploración, también reducido. explore_ds: ↑ más fluidez, ↓2 más fino.
    int explore_ds <- 4;
    int explore_cols <- max(1, int(grid_cols / explore_ds));
    int explore_rows <- max(1, int(grid_rows / explore_ds));
    matrix exploration_matrix <- 0.0 as_matrix {explore_cols, explore_rows};
    field exploration_field <- field(exploration_matrix);

    // Redefinir el mundo al tamaño del heatmap. Como atributo (no en init)
    // para evitar el warning de cambio dinámico de shape del mundo.
    geometry shape <- rectangle(grid_cols, grid_rows) translated_by {grid_cols / 2.0, grid_rows / 2.0};

    // --- Enlaces de comunicación entre agentes (refrescados cada tick) ---
    list<list<point>> comm_links <- [];

    // --- Estado de la simulación ---
    int current_step <- 0;
    int victims_found <- 0;
    int victims_total <- 0;
    bool simulation_ended <- false;
    bool init_done <- false;
    bool connected_to_python <- false;
    string last_status <- "Pulsa Play para conectar a Python...";
    // Buffer TCP: solo se procesan comandos completos (terminados en '~');
    // el fragmento final sin terminador se guarda para el siguiente ciclo.
    string rx_buffer <- "";
    // Control de flujo: GAMA envía "ACK|commands_processed" para que Python
    // no envíe más rápido de lo que GAMA consume (si no, su cola de red descarta).
    int commands_processed <- 0;
    int last_acked_count <- -1;

    init {
        // Rellenar el heatmap de render muestreando prob_matrix (1 de cada
        // prob_ds celdas). Una sola vez al cargar el modelo, antes de Play.
        loop ix from: 0 to: disp_cols - 1 {
            loop iy from: 0 to: disp_rows - 1 {
                prob_display_matrix[{ix, iy}] <- float(prob_matrix[{ix * prob_ds, iy * prob_ds}]);
            }
        }
        probability_field <- field(prob_display_matrix);

        // La conexión TCP se hace en el primer cycle (reflex connect_to_python).
        write "** SAR Network — Listo. Pulsa Play para conectar a " + python_host + ":" + string(python_port);
    }

    // Handshake: al pulsar Play arrancan los reflexes y se conecta a Python,
    // que entonces desbloquea su wait_for_gama() y envía INIT.
    reflex connect_to_python when: !connected_to_python {
        write "** Conectando a Python en " + python_host + ":" + string(python_port) + "...";
        do connect to: python_host protocol: "tcp_client" port: python_port raw: true with_name: "python";
        connected_to_python <- true;
        last_status <- "Conectado. Esperando init...";
        write "** Conectado.";
    }

    // --- Reflex: leer y procesar mensajes TCP cada ciclo ---
    reflex fetch_data when: !simulation_ended {
        // 1) Drenar el mailbox al buffer (un mensaje puede traer varios comandos
        //    o un fragmento). Guardamos el remitente para devolverle el ACK.
        message last_msg <- nil;
        loop while: has_more_message() {
            message msg <- fetch_message();
            rx_buffer <- rx_buffer + string(msg.contents);
            last_msg <- msg;
        }

        // 2) Procesar solo comandos completos (terminados en '~'); el trozo
        //    incompleto final se conserva para el próximo ciclo.
        if (rx_buffer != "" and (rx_buffer contains "~")) {
            bool ends_complete <- copy_between(rx_buffer, length(rx_buffer) - 1, length(rx_buffer)) = "~";
            list<string> tokens <- rx_buffer split_with "~";
            int n <- length(tokens);
            int process_count <- ends_complete ? n : (n - 1);
            if (process_count > 0) {
                loop i from: 0 to: process_count - 1 {
                    string line <- tokens[i];
                    line <- line replace("\n", "");
                    line <- line replace("\r", "");
                    if (length(line) > 0) {
                        commands_processed <- commands_processed + 1;
                        do process_line(line);
                    }
                }
            }
            // Conservar el fragmento incompleto (o vaciar si todo era completo).
            rx_buffer <- ends_complete ? "" : tokens[n - 1];
        }

        // 3) Control de flujo: confirmar a Python cuántos comandos llevamos
        //    procesados, para que no se adelante y sature su cola de red.
        if (last_msg != nil and commands_processed != last_acked_count) {
            do send to: last_msg.sender contents: ("ACK|" + string(commands_processed) + "~");
            last_acked_count <- commands_processed;
        }
    }

    // Reflex: actualizar exploration field periódicamente
    reflex update_exploration when: init_done and mod(cycle, 100) = 0 {
        exploration_field <- field(exploration_matrix);
    }

    action process_line(string line) {
        list<string> parts <- line split_with "|";
        string cmd <- parts[0];

        if (cmd = "INIT" and length(parts) >= 6) {
            victims_total <- int(float(parts[3]));
            write "INIT recibido: " + parts[1] + " drones, " + parts[2] + " dogs, " + parts[3] + " victimas, grid=" + parts[4] + "x" + parts[5];
            last_status <- "Inicializando...";
        }
        else if (cmd = "DRONE" and length(parts) >= 5) {
            int didx <- int(parts[1]);
            create drone {
                agent_idx <- didx;
                location <- {float(parts[2]), float(parts[3])};
                budget_left <- float(parts[4]);
                is_active <- true;
            }
        }
        else if (cmd = "DOG" and length(parts) >= 5) {
            int didx <- int(parts[1]);
            create robot_dog {
                agent_idx <- didx;
                location <- {float(parts[2]), float(parts[3])};
                budget_left <- float(parts[4]);
                is_active <- true;
            }
        }
        else if (cmd = "VICTIM" and length(parts) >= 4) {
            // Idempotente por índice: Python reenvía las víctimas para rellenar
            // las que GAMA descartó en la ráfaga de init; no se duplican.
            int vidx <- int(parts[1]);            if (empty(victim where (each.v_idx = vidx))) {
                create victim {
                    v_idx <- vidx;
                    location <- {float(parts[2]), float(parts[3])};
                    found <- false;
                }
            }
        }
        else if (cmd = "INIT_END") {
            init_done <- true;
            write "Init completado: " + length(drone) + " drones, " + length(robot_dog) + " dogs, " + length(victim) + " victimas";
            last_status <- "Simulacion en curso...";
        }
        else if (cmd = "TICK" and length(parts) >= 2) {
            current_step <- int(float(parts[1]));
        }
        else if (cmd = "AGENT" and length(parts) >= 7) {
            string atype <- parts[1];
            int aidx <- int(float(parts[2]));
            float ax <- float(parts[3]);
            float ay <- float(parts[4]);
            float abudget <- float(parts[5]);
            bool aactive <- int(parts[6]) > 0;

            if (atype = "drone") {
                drone target_drone <- first(drone where (each.agent_idx = aidx));
                if (target_drone != nil) {
                    ask target_drone {
                        do move_to(ax, ay, abudget, aactive);
                    }
                } else {
                    // Red de seguridad: si el DRONE de init se perdió, crearlo aquí.
                    create drone {
                        agent_idx <- aidx;
                        location <- {ax, ay};
                        budget_left <- abudget;
                        is_active <- aactive;
                    }
                }
                int cx <- min(explore_cols - 1, max(0, int(ax / explore_ds)));
                int cy <- min(explore_rows - 1, max(0, int(ay / explore_ds)));
                exploration_matrix[{cx, cy}] <- 1.0;
            } else if (atype = "robot_dog") {
                robot_dog target_dog <- first(robot_dog where (each.agent_idx = aidx));
                if (target_dog != nil) {
                    ask target_dog {
                        do move_to(ax, ay, abudget, aactive);
                    }
                } else {
                    // Red de seguridad: si el DOG de init se perdió, crearlo aquí.
                    create robot_dog {
                        agent_idx <- aidx;
                        location <- {ax, ay};
                        budget_left <- abudget;
                        is_active <- aactive;
                    }
                }
                int cx <- min(explore_cols - 1, max(0, int(ax / explore_ds)));
                int cy <- min(explore_rows - 1, max(0, int(ay / explore_ds)));
                exploration_matrix[{cx, cy}] <- 1.0;
            }
        }
        else if (cmd = "FOUND" and length(parts) >= 3) {
            float vx <- float(parts[1]);
            float vy <- float(parts[2]);
            // Encontrar la víctima no-encontrada más cercana
            list<victim> unfound <- victim where (!each.found);
            if (length(unfound) > 0) {
                victim closest_v <- unfound closest_to {vx, vy};
                if (closest_v != nil) {
                    ask closest_v { do set_found; }
                }
            }
            victims_found <- length(victim where (each.found));
        }
        else if (cmd = "TICK_END") {
            // Tick procesado
        }
        else if (cmd = "LINKS" and length(parts) >= 2) {
            // Formato: LINKS|n|x1,y1,x2,y2;x3,y3,x4,y4;...
            comm_links <- [];
            if (length(parts) >= 3 and length(parts[2]) > 0) {
                list<string> segs <- parts[2] split_with ";";
                loop s over: segs {
                    list<string> nums <- s split_with ",";
                    if (length(nums) = 4) {
                        comm_links <- comm_links + [[
                            {float(nums[0]), float(nums[1])},
                            {float(nums[2]), float(nums[3])}
                        ]];
                    }
                }
            }
        }
        else if (cmd = "END") {
            simulation_ended <- true;
            last_status <- "Simulacion finalizada.";
            write "Simulacion finalizada.";
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  SPECIES
// ─────────────────────────────────────────────────────────────────

species drone {
    int agent_idx <- -1;
    float budget_left;
    bool is_active <- true;
    list<point> trail <- [];

    action move_to (float x, float y, float budget, bool active) {
        if (is_active) {
            trail <- trail + [location];
            if (length(trail) > 200) {
                trail <- trail copy_between(length(trail) - 200, length(trail));
            }
        }
        location <- {x, y};
        budget_left <- budget;
        is_active <- active;
    }

    aspect default {
        if (is_active) {
            // Drone: triángulo cyan (los dogs son cuadrados; la forma los distingue).
            draw triangle(7) color: #cyan border: #white;
        } else {
            // Drone inactivo (presupuesto agotado): triángulo gris visible
            draw triangle(6) color: rgb(40, 80, 90, 220) border: rgb(0, 180, 200, 180);
        }

        if (length(trail) > 1) {
            draw polyline(trail) color: rgb(0, 200, 230, 180) width: 0.4;
        }
    }
}

species robot_dog {
    int agent_idx <- -1;
    float budget_left;
    bool is_active <- true;
    list<point> trail <- [];

    action move_to (float x, float y, float budget, bool active) {
        if (is_active) {
            trail <- trail + [location];
            if (length(trail) > 200) {
                trail <- trail copy_between(length(trail) - 200, length(trail));
            }
        }
        location <- {x, y};
        budget_left <- budget;
        is_active <- active;
    }

    aspect default {
        if (is_active) {
            draw square(8) color: #cyan border: #white;
        } else {
            // Dog inactivo: cuadrado gris oscuro con borde visible
            draw square(7) color: rgb(40, 80, 90, 220) border: rgb(0, 180, 200, 180);
        }

        if (length(trail) > 1) {
            draw polyline(trail) color: rgb(0, 200, 230, 180) width: 0.4;
        }
    }
}

species victim {
    int v_idx <- -1;        // índice global (para reenvío idempotente)
    bool found <- false;

    action set_found {
        found <- true;
    }

    aspect default {
        if (found) {
            draw circle(5) color: #green;
            draw "V" at: location + {0, -6} color: #green font: font("Arial", 10, #bold);
        } else {
            draw circle(4) color: #yellow;
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  EXPERIMENT
// ─────────────────────────────────────────────────────────────────

experiment sar_gui_network type: gui {
    parameter "Python host" var: python_host;
    parameter "Python port" var: python_port;

    output synchronized: false {
        display main type: opengl background: #black {
            // Heatmap de probabilidad: estático y reducido (prob_ds). Se dibuja
            // una sola vez ('refresh: false'), no se re-renderiza por frame.
            mesh probability_field
                color: palette([#black, #blue, #yellow, #orange, #red])
                transparency: 0.3
                smooth: false
                refresh: false
                scale: 0;

            // Overlay de zonas exploradas (verde): cambia cada tick, se refresca.
            mesh exploration_field
                color: palette([rgb(0,0,0,0), rgb(0, 255, 0, 150)])
                transparency: 0.4
                scale: 0;

            // Agentes (encima de los mesh)
            species victim;
            species drone;
            species robot_dog;

            // Enlaces de comunicación activos (líneas cyan entre agentes en rango)
            graphics "comm_links" {
                loop link over: comm_links {
                    draw line(link) color: rgb(80, 220, 255, 180) width: 0.6;
                }
            }
        }

        // Panel de métricas
        display metrics type: image refresh: every(5 #cycles) {
            chart "Victimas encontradas" type: series {
                data "Encontradas" value: victims_found color: #green;
                data "Total" value: victims_total color: #yellow;
            }
            chart "Budget restante (m)" type: series {
                loop d over: drone {
                    data d.name value: d.budget_left;
                }
                loop dog over: robot_dog {
                    data dog.name value: dog.budget_left;
                }
            }
        }

        // Monitores de estado
        monitor "Step" value: current_step;
        monitor "Victimas" value: "" + victims_found + " / " + victims_total;
        monitor "Estado" value: last_status;
    }
}
