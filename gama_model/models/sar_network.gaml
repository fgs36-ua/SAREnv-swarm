/**
 * SAR Network — Visualización 3D en GAMA GUI con datos en tiempo real.
 *
 * Abre este modelo en GAMA Platform (GUI) y ejecuta el experimento.
 * Se conecta como cliente TCP a un servidor Python que ejecuta la simulación.
 * Python controla toda la lógica; GAMA solo renderiza y anima.
 *
 * Uso:
 *   1. python examples/14_gama_gui_visualization.py --scenario 1
 *   2. Doble-click en "Experiment sar_gui_network" en GAMA
 *   3. Pulsa ▶ (Play) en la barra de herramientas de GAMA
 *
 * Protocolo TCP (campos separados por |, cada comando termina en ~\n):
 *   INIT|nd|ndog|nv|cols|rows
 *   DRONE|idx|x|y|budget   DOG|idx|x|y|budget   VICTIM|idx|x|y
 *   INIT_END   TICK|step   AGENT|type|idx|x|y|budget|active
 *   FOUND|x|y   TICK_END   PHEROMONE   END
 */
model SARNetwork

global skills: [network] {
    // --- Parámetros configurables desde la GUI ---
    string python_host <- "localhost";
    int python_port <- 6869;

    // --- Datos estáticos ---
    file heatmap_file <- csv_file("../includes/heatmap.csv");

    // --- Fields para overlay ---
    matrix prob_matrix <- matrix(heatmap_file);
    field probability_field <- field(prob_matrix);
    int grid_cols <- prob_matrix.columns;
    int grid_rows <- prob_matrix.rows;
    matrix exploration_matrix <- 0.0 as_matrix {grid_cols, grid_rows};
    field exploration_field <- field(exploration_matrix);

    // Redefinir el mundo al tamaño del heatmap (cols × rows). Hacerlo
    // como atributo (no en init) evita el warning "Dynamically changing
    // the shape of the world can lead to unexpected results".
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

    init {
        // NOTA: la conexión TCP se hace en el primer cycle (reflex
        // connect_to_python), no aquí. Esto actúa como handshake:
        // Python no recibe nada hasta que el usuario pulse Play.
        write "** SAR Network v8 — Listo. Pulsa Play para conectar a " + python_host + ":" + string(python_port);
    }

    // Handshake: la conexión TCP se establece al pulsar Play (cuando
    // los reflexes empiezan a correr). Python desbloquea su
    // wait_for_gama() en ese momento y envía INIT.
    reflex connect_to_python when: !connected_to_python {
        write "** Conectando a Python en " + python_host + ":" + string(python_port) + "...";
        do connect to: python_host protocol: "tcp_client" port: python_port raw: true with_name: "python";
        connected_to_python <- true;
        last_status <- "Conectado. Esperando init...";
        write "** v8 Conectado.";
    }

    // --- Reflex: leer y procesar mensajes TCP cada ciclo ---
    reflex fetch_data when: !simulation_ended {
        // Debug heartbeat
        if (mod(cycle, 5000) = 0) {
            write "[v8] cycle=" + cycle + " mbox=" + length(mailbox) + " init=" + init_done + " step=" + current_step;
        }

        // Cada mensaje del mailbox puede contener varias líneas
        // concatenadas por TCP coalescing (GAMA raw TCP elimina \n).
        // El delimitador ~ marca el fin de cada comando.
        loop while: has_more_message() {
            message msg <- fetch_message();
            string raw <- string(msg.contents);
            list<string> lines <- raw split_with "~";
            loop line over: lines {
                line <- line replace("\n", "");
                line <- line replace("\r", "");
                if (length(line) > 0) {
                    do process_line(line);
                }
            }
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
            create victim {
                location <- {float(parts[2]), float(parts[3])};
                found <- false;
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
                    // Drone no existe aún (mensaje de INIT perdido): crearlo on-the-fly
                    write "[WARN] Drone " + aidx + " no encontrado, creando on-the-fly en (" + ax + "," + ay + ")";
                    create drone {
                        agent_idx <- aidx;
                        location <- {ax, ay};
                        budget_left <- abudget;
                        is_active <- aactive;
                    }
                }
                int cx <- min(grid_cols - 1, max(0, int(ax)));
                int cy <- min(grid_rows - 1, max(0, int(ay)));
                exploration_matrix[{cx, cy}] <- 1.0;
            } else if (atype = "robot_dog") {
                robot_dog target_dog <- first(robot_dog where (each.agent_idx = aidx));
                if (target_dog != nil) {
                    ask target_dog {
                        do move_to(ax, ay, abudget, aactive);
                    }
                } else {
                    write "[WARN] Dog " + aidx + " no encontrado, creando on-the-fly en (" + ax + "," + ay + ")";
                    create robot_dog {
                        agent_idx <- aidx;
                        location <- {ax, ay};
                        budget_left <- abudget;
                        is_active <- aactive;
                    }
                }
                int cx <- min(grid_cols - 1, max(0, int(ax)));
                int cy <- min(grid_rows - 1, max(0, int(ay)));
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
        else if (cmd = "PHEROMONE") {
            // Actualizar field desde la matriz local
            exploration_field <- field(exploration_matrix);
        }
        else if (cmd = "GOSSIP_DATA" and length(parts) >= 4) {
            // [DEPRECADO] El campo gossip ya no se renderiza como mesh
            // (ocultaba a los agentes). Se ignora silenciosamente para
            // mantener compatibilidad con servers viejos.
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
            // Drone: triángulo cyan (los dogs son cuadrados cyan; la forma
            // basta para diferenciarlos sin necesidad de color distinto).
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
            // Heatmap de probabilidad (plano, sin elevación 3D)
            mesh probability_field
                color: palette([#black, #blue, #yellow, #orange, #red])
                transparency: 0.3
                smooth: true
                scale: 0;

            // Overlay: zonas exploradas (verde)
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
