COMPLEXITY_LEVELS = [
    {
        "id": 0,
        "title": "Level 0: Static sites",
        "description": "Простые сайты без JS или только с HTML-формами",
        "criteria": ["basic_html"],
    },
    {
        "id": 1,
        "title": "Level 1: Simple JS",
        "description": "Фронтовая валидация, inline-скрипты, onsubmit/onchange",
        "criteria": ["SimpleJS"],
    },
    {
        "id": 2,
        "title": "Level 2: jQuery / Vanilla AJAX",
        "description": "Использование jQuery, fetch(), axios(), AJAX-вызовы, DOM-манипуляции",
        "criteria": ["jQuery", "Ajax"],
    },
    {
        "id": 3,
        "title": "Level 3: Bootstrap / UI components",
        "description": "Использование Bootstrap, модальных окон, табов, кастомных UI-компонентов",
        "criteria": ["Bootstrap", "UI_components"],
    },
    {
        "id": 4,
        "title": "Level 4: Single Page Application (SPA)",
        "description": "Vue, React, Angular, роутинг, клиентский рендеринг, lazy-load",
        "criteria": ["Vue", "React", "Angular", "SPA"],
    },
    {
        "id": 5,
        "title": "Level 5: CMS / E-commerce / CRM",
        "description": "WordPress, Joomla, Drupal, Laravel, Yii2, Magento, OpenCart, 1C-Bitrix",
        "criteria": [
            "WordPress",
            "Joomla",
            "Drupal",
            "Laravel",
            "Yii2",
            "Magento",
            "OpenCart",
            "Bitrix",
        ],
    },
    {
        "id": 6,
        "title": "Level 6: Custom engines / API-heavy",
        "description": "REST/GraphQL, PWA, WebSocket, кастомные движки",
        "criteria": ["GraphQL", "websocket_heavy", "pwa_capable", "custom_logic"],
    },
    {
        "id": 7,
        "title": "Level 7: Heavy apps (Realtime)",
        "description": "Google Docs, Miro, Figma, коллаборативные редакторы, real-time sync",
        "criteria": ["collaborative", "realtime", "RTC", "client_storage"],
    },
]
