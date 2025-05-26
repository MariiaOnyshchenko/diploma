import csv
import os
import time
import logging
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Налаштування Selenium WebDriver
def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument('--headless=new')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.set_page_load_timeout(60)
    return driver

# Запис порції відгуків у CSV
def save_reviews_chunk(reviews, filenames):
    for filename in filenames:
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['review', 'stars'])  # Заголовки
            for review in reviews:
                writer.writerow(review)
        logging.info(f'Дозаписано {len(reviews)} відгуків до {filename}')

# Основна логіка
def main():
    base_url = 'https://www.vidhuk.ua/uk/internet-magazin-makeupcomua.html?page={}'
    filenames = ['vidhuk_reviews.csv', 'vidhuk_reviews_new_makeup.csv']
    total_reviews = 0
    page_number = 1
    driver = setup_driver()

    while True:
        url = base_url.format(page_number)
        logging.info(f'Відкриваю сторінку: {url}')
        try:
            driver.get(url)
        except TimeoutException:
            logging.warning(f'Таймаут при завантаженні сторінки: {url}')
            break

        time.sleep(2)

        try:
            review_elements = driver.find_elements(By.CSS_SELECTOR, '.review-snippet')
            star_elements = driver.find_elements(By.CSS_SELECTOR, '.star_ring span[style*="width"]')
        except Exception as e:
            logging.error(f'Помилка при пошуку елементів: {e}')
            break

        if not review_elements:
            logging.info(f'На сторінці {page_number} відгуків не знайдено. Завершення.')
            break

        if len(review_elements) != len(star_elements):
            logging.warning(f'Кількість відгуків ({len(review_elements)}) і зірок ({len(star_elements)}) не збігається')

        page_reviews = []
        for idx, el in enumerate(review_elements):
            try:
                review_text = el.text.strip()
                if not review_text:
                    continue

                # Обробка зірок по індексу
                try:
                    style = star_elements[idx].get_attribute('style')
                    match = re.search(r'width\s*:\s*(\d+)\s*px', style)
                    stars = int(match.group(1)) // 13 if match else 0
                except Exception as e:
                    logging.warning(f'Зірки не знайдено для відгуку #{idx}: {e}')
                    stars = 0

                # Фільтрація українських відгуків
                if re.search(r'[ЄєІіЇїҐґ]', review_text) and not re.search(r'[ыЭэъЁё]', review_text):
                    page_reviews.append([review_text, stars])

            except Exception as e:
                logging.warning(f'Помилка при обробці відгуку #{idx}: {e}')

        if page_reviews:
            save_reviews_chunk(page_reviews, filenames)
            total_reviews += len(page_reviews)
            if total_reviews % 100 < len(page_reviews):
                logging.info(f'Усього зчитано: {total_reviews} відгуків')

        page_number += 1

    driver.quit()
    logging.info(f'Готово. Загалом зчитано {total_reviews} відгуків.')

if __name__ == '__main__':
    main()
