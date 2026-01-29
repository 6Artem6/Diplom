FRAMEWORK_PATTERNS = {
    # Frontend
    "React": [r"data-reactroot", r"__REACT_DEVTOOLS_GLOBAL_HOOK__"],
    "Vue": [r"__VUE_DEVTOOLS_GLOBAL_HOOK__", r"\bv-", r"router-view"],
    "Angular": [r"ng-version", r"platform-browser", r"ng-app"],
    "Svelte": [r"svelte%-[a-z0-9]+", r"__SVELTE_DEVTOOLS_GLOBAL_HOOK__"],
    "Next.js": [r"_next/static", r"next-route-announcer"],
    "Nuxt.js": [r"nuxt%-link", r"__NUXT__"],
    "Ember": [r"ember%-application", r"ember%-view"],
    "Backbone": [r"Backbone\.Model", r"Backbone\.View"],
    # Backend CMS / Frameworks
    "Yii2": [r"yiiActiveForm", r"yiiGridView"],
    "Laravel": [r"XSRF%-TOKEN", r"laravel_session"],
    "Symfony": [r"symfony%-profiler%-toolbar", r"_profiler/empty"],
    "CodeIgniter": [r"ci_session", r"CodeIgniter"],
    "Zend": [r"Zend_Version", r"Zend_Controller"],
    "Django": [r"csrftoken", r"django%-admin"],
    "Flask": [r"flask_session", r"flask_wtf"],
    "Rails": [r"csrf%-token", r"rails%-csrf"],
    "Spring": [r"JSESSIONID", r"spring%-security%-csrf"],
    "Express": [r"express_session", r"x%-powered%-by: express"],
    # CMS
    "WordPress": [r"wp%-content", r"wp%-json"],
    "Joomla": [r"Joomla!", r"com_content"],
    "Drupal": [r"Drupal\.settings", r"drupal%-settings%-json"],
    "Magento": [r"mage%-cookies", r"Magento_Ui"],
    "Shopify": [r"cdn\.shopify\.com", r"Shopify\.theme"],
    "Bitrix": [r"bitrix%-sessid", r"bitrix%-panel"],
    "OpenCart": [r"index\.php%?route=", r"opencart"],
    # UI Libraries
    "jQuery": [r"jQuery", r"\$\("],
    "Bootstrap": [r"bootstrap\.js", r"data%-bs%-toggle", r"modal fade"],
    "Tailwind": [r"tailwind%-config", r"tailwindcss"],
    "MaterialUI": [r"MuiButton%-root", r"MuiSvgIcon%-root"],
}

JS_COMPLEXITY_PATTERNS = {
    # Simple inline JS
    "SimpleJS": [r"onsubmit", r"onchange", r"oninput"],
    # Ajax-based pages
    "Ajax": [r"\.ajax", r"fetch\(", r"axios\("],
    # SPA frameworks
    "SPA": [r"router%-view", r"ng%-router", r"ReactDOM\.render"],
    # GraphQL-driven apps
    "GraphQL": [r"graphql", r"/graphql"],
    # WebSockets / realtime apps
    "Realtime": [r"new WebSocket", r"socket\.io", r"Pusher"],
    # Micro-frontend or bundlers
    "Webpack": [r"webpackBootstrap", r"__webpack_require__"],
    "Vite": [r"vite%-hot%-update", r"import\.meta\.env"],
}
