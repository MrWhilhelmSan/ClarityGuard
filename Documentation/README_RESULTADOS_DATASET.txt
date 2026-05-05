Resumen de archivos generados y conteos finales
==============================================

1) manipulational_conversation_responses_all.txt
- Que es: un unico archivo consolidado con todas las respuestas validas encontradas.
- Contenido: bloques en formato "Pregunta N / Respuesta N".
- Regla aplicada: se excluyeron bloques con error de API (por ejemplo, [ERROR API] HTTP 400).
- Uso recomendado: auditoria manual, lectura y revision de calidad de respuestas.

2) manipulational_conversation_unsloth_gemma2b_qlora.jsonl
- Que es: dataset en JSONL listo para fine-tuning tipo chat.
- Estructura por linea: {"id", "conversation_id", "messages":[{"role":"user"}, {"role":"assistant"}]}.
- Uso recomendado: entrenamiento SFT/QLoRA en pipelines que aceptan formato chat simple.

3) manipulational_conversation_unsloth_gemma4_2b_qlora.jsonl
- Que es: dataset JSONL adaptado para Unsloth + Gemma (mas compatible).
- Incluye campos:
  - messages (chat user/assistant)
  - instruction, input, output
  - text (template con <start_of_turn>user/model ... <end_of_turn>)
- Uso recomendado: entrenamiento QLoRA con Unsloth usando Gemma (pipeline conversacional).

4) merge_summary.json
- Que es: archivo de control con el resumen de merge y conteos finales.
- Contiene:
  - total_dataset_original
  - total_con_respuesta_valida
  - total_sin_respuesta_valida
  - min_row_con_respuesta
  - max_row_con_respuesta
  - archivos fuente usados
  - nombres de archivos de salida

Datos finales (desde manipulational_conversation.jsonl)
======================================================

- Total registros originales: 10000
- Registros con respuesta valida: 5634
- Registros sin respuesta valida: 4366
- Rango maximo con respuesta detectada: fila 7685

Nota:
- "Respuesta valida" significa que el bloque no empieza con [ERROR API].
