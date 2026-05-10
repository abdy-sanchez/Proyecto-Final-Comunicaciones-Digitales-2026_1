# **CONTEXTO**

El proyecto consiste en diseñar e implementar un **módem óptico espacio-temporal half-duplex** que permita transmitir información entre dos computadores sin usar una conexión de red. La idea principal es que el transmisor convierta un archivo de texto en una secuencia de cuadros visuales mostrados en pantalla, mientras que el receptor capture esos cuadros con una cámara web, corrija las distorsiones del canal óptico y reconstruya el mensaje original.

El objetivo final es lograr la transmisión de un texto de **500 caracteres** desde la pantalla de un computador hacia la cámara de otro, a una distancia mínima de **50 cm**, en un tiempo máximo de **10 segundos**, manteniendo una baja tasa de error. Para ello, el sistema debe integrar técnicas propias de comunicaciones digitales, como modulación, sincronización, detección, calibración, corrección geométrica y codificación de canal, adaptadas al entorno práctico pantalla–cámara.
