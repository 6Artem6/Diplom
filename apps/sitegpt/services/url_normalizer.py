import re
import urllib.parse


class URLNormalizer:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.parsed_base = urllib.parse.urlparse(base_url)
        self.domain = self.parsed_base.netloc

    def _same_domain(self, url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        return parsed.netloc == "" or parsed.netloc.endswith(self.domain)

    def normalize(self, url: str, current_url: str) -> str | None:
        """
        Универсальная нормализация URL:
        - Любые query-параметры → {val}
        - Числа в пути → {id}, UUID → {uuid}, остальное → {val}
        """
        abs_url = urllib.parse.urljoin(current_url, url.split("#")[0])
        parsed = urllib.parse.urlparse(abs_url)

        if not self._same_domain(abs_url):
            return None

        # нормализуем path
        segments = parsed.path.split("/")
        normalized_segments = []
        for seg in segments:
            if not seg:
                continue
            # числа → {id}
            elif re.fullmatch(r"\d+", seg):
                normalized_segments.append("{id}")
            # UUID → {uuid}
            elif re.fullmatch(r"[0-9a-fA-F-]{8,}", seg):
                normalized_segments.append("{uuid}")
            # текст → {val}
            else:
                normalized_segments.append("{val}")

        path_norm = "/" + "/".join(normalized_segments)

        # query параметры → {val}
        query_dict = urllib.parse.parse_qs(parsed.query)
        if query_dict:
            query_norm = "&".join(f"{k}={{val}}" for k in sorted(query_dict.keys()))
            path_norm += f"?{query_norm}"

        return path_norm or "/"

    def is_new(self, url: str) -> bool:
        """
        Проверка, новый ли маршрут (по нормализованному паттерну).
        """
        pattern = self.normalize(url)
        if pattern in self.seen_patterns:
            return False
        self.seen_patterns.add(pattern)
        return True
