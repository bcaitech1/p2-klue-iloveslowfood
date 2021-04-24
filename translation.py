import argparse
from copy import deepcopy
import os
import time
from tqdm import tqdm
import json
from multiprocessing import Pool, Manager
from itertools import repeat
import numpy as np
from selenium import webdriver


class Translator(object):
    def __init__(self, input_dir, output_dir, multiprocessor=1, path="chromedriver"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.multiprocessor = multiprocessor
        self.cpath = path
        manager = Manager()
        self.translated = manager.dict()

        # Load not translated words
        if os.path.isfile(output_dir):
            self.translated = manager.dict(self.load_json(output_dir))
            tmp = self.load_text(input_dir)
            corpus = []
            for i in range(len(tmp)):
                if tmp[i].strip() not in self.translated:
                    corpus.append(tmp[i].strip())
            self.n = len(corpus)
            print(
                f"You got file of {output_dir} \n{len(tmp) - len(corpus)} lines translated"
            )
            print(f"You have left {self.n} of total {len(tmp)}")

        else:
            corpus = self.load_text(input_dir)
            self.n = len(corpus)
            print(f"You have left {self.n}")

        self.parts = np.array_split(corpus, self.multiprocessor)

    def kr2en(self, sentence, driver, time_delay):
        input_box = driver.find_element_by_css_selector("#sourceEditArea textarea")
        input_box.clear()
        input_box.clear()
        input_box.clear()
        input_box.clear()
        input_box.send_keys(sentence)
        driver.find_element_by_css_selector("#btnTranslate").click()
        time.sleep(time_delay)
        result = str(driver.find_element_by_css_selector("#txtTarget").text)
        input_box.clear()
        input_box.clear()
        input_box.clear()
        input_box.clear()
        return result

    def set_language(self, driver, trans_type: str = "eng"):
        print(f"Set languages: KOR -> {trans_type}...", sep="\t")
        to_language = driver.find_element_by_css_selector(
            "#ddTargetLanguage2 > div.dropdown_top___13QlJ > button:nth-child(2)"
        )
        to_language.click()
        print('click!')
        driver.implicitly_wait(10)
#         ActionChains(driver).move_to_element(to_language).click(to_language).perform()

        eng = driver.find_element_by_css_selector(
            "#ddTargetLanguage2 > div.dropdown_menu___XsI_h.active___3VPGL > ul > li:nth-child(2)"
        )
        eng.click()
        driver.implicitly_wait(10)
#         ActionChains(driver).move_to_element(eng).click(eng).perform()
        print("ENG selected", sep="\t")

        print(f"done!")

    def translate(self):
        if self.n != 0:
            pool = Pool(self.multiprocessor)
            pool.starmap(self._translate, zip(self.parts, repeat(self.translated)))
            pool.close()
            pool.join()
            self.save_json(self.output_dir, self.translated._getvalue())
        print("file translation completed")
        os.system("pkill chromium")
        os.system("pkill chrome")
        return self.translated._getvalue()

    def _translate(self, corpus, translated):
        not_translated = []
        prev_result = result = ""

        for idx, sentence in tqdm(enumerate(corpus)):
            if idx % 500 == 0:
                driver = self.init_driver()
                print("driver initialized")
                self.save_json(self.output_dir, self.translated._getvalue())

            sentence = sentence.strip()
            time_delay = 2
            flag = False
            while (
                result == prev_result and sentence not in translated and time_delay < 10
            ):
                if time_delay > 3:
                    print(time_delay)
                    print(prev_sentence, sentence)
                    print(prev_result, result)

                result = self.kr2en(sentence, driver, time_delay)
                time_delay += 1

            # when sentence is not translated
            if result == "":
                print(f'"{sentence}" \t fail to translate')
                not_translated.append(sentence)
                continue

            prev_result = deepcopy(result)
            prev_sentence = deepcopy(sentence)

            translated[sentence] = result.strip()

        if not_translated != []:
            print(f"There are some sentences not translated : {not_translated}")
            return self._translate(self, not_translated, translated)

        return translated

    def load_json(self, json_dir):
        with open(json_dir, encoding="UTF8") as f:
            json_file = json.load(f)
        print(f"{json_dir} loaded")
        return json_file

    def save_json(self, save_dir, result_dict):
        with open(save_dir, "w", encoding="UTF8") as f:
            json.dump(result_dict, f, ensure_ascii=False)

    def load_text(self, text_dir):
        with open(text_dir, encoding="UTF8") as f:
            text_file = f.readlines()
        print(f"{text_dir} loaded")
        return text_file

    def init_driver(self, cpath=None):
        if cpath == None:
            cpath = self.cpath
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.3538.102 Safari/537.36"
        )
        options.add_argument("--start-maximized")
        driver = webdriver.Chrome(executable_path=cpath, options=options)
        driver.implicitly_wait(15)
        driver.get("https://papago.naver.com/")

        return driver


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Before get started you need read README"
    )

    parser.add_argument("--input_dir", type=str, default="./preprocessed/test_entity_korean.txt")
    parser.add_argument("--output_dir", type=str, default="./preprocessed/test_entity_kr2en.json")
    parser.add_argument("--multiprocessor", type=int, default=int(8))
    parser.add_argument("--path", type=str, default="chromedriver")
    args = parser.parse_args()
    translator = Translator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        multiprocessor=args.multiprocessor,
        path=args.path,
    )

    translator.translate()
